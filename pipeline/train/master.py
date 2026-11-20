"""全量微调时语言模型参数的两种更新方式（任务 #66，选项见 precision.py）。

模型中语言模型的参数保持 bfloat16，前向和反传都用它们计算，计算方式与只训练 projector 时相同。

MasterWeights（--language-model-master-weights fp32，默认）：每个 bfloat16 参数另有一份 float32 主权重，
优化器只更新主权重：

1. 反传中某个参数的梯度一累加到它的 .grad 上，钩子（register_post_accumulate_grad_hook）就把它转成梯度精度
   （--language-model-grad-dtype，默认 float32）加到主权重的梯度上，并释放 bfloat16 的梯度。梯度累积的各个微批在
   这个精度中相加；与输出层共用的词嵌入从输出层和词嵌入两处得到的梯度由 autograd 先相加，再交给钩子一次。
   梯度精度为 bf16 时主权重的 grad_dtype 设为 bfloat16，梯度原样移过去，不转换。
2. 裁剪梯度、AdamW 更新都作用在主权重上。
3. 更新后把主权重舍入成 bfloat16 写回模型参数（copy_to_model），下一步的前向用新的权重。

DirectWeights（--language-model-master-weights none）：优化器直接更新模型中的 bfloat16 参数，没有主权重。
梯度精度为 fp32 时把参数的 grad_dtype 设为 float32，autograd 把反传得到的梯度转成 float32 再累加到 .grad 上；
bf16 时梯度保持 bfloat16。

显存（每个参数）：模型中的 bfloat16 参数 2 字节（只训练 projector 时就已经在显存中），主权重 4 字节（只有
MasterWeights），梯度 4 或 2 字节，AdamW 状态见 precision.py。模型中本来就是 float32 的参数两种方式下都直接
交给优化器，梯度也保持 float32。

续训：state_dict 给出优化器更新的那份权重（主权重，或模型参数本身），随 trainer_state.pt 保存在 STATE_KEY 下；
续训时把它们交给构造函数，写回模型参数，模型中的 bfloat16 参数与保存时逐位相同。
"""

import torch


def check_state(named_parameters, state, what):
    """续训时核对检查点中的权重：键与模型一致，形状相同。"""
    expected = {name for name, _ in named_parameters}
    missing = sorted(expected - set(state))
    extra = sorted(set(state) - expected)
    if missing or extra:
        raise ValueError(f"检查点中语言模型{what}的键与模型不一致。缺少：{missing[:5]}；多余：{extra[:5]}")
    for name, parameter in named_parameters:
        if tuple(state[name].shape) != tuple(parameter.shape):
            raise ValueError(f"{name} 形状为 {tuple(state[name].shape)}，模型中为 {tuple(parameter.shape)}")


class MasterWeights:
    """语言模型参数的 float32 主权重，以及把梯度转入主权重、把更新后的主权重写回模型的钩子。"""

    # 续训状态 trainer_state.pt 中保存主权重的键。
    STATE_KEY = "language_model_master"

    def __init__(self, named_parameters, state = None, grad_dtype = torch.float32):
        """named_parameters 为 [(名称, 参数)]，都需要梯度。

        state 为续训时检查点中的 {名称: float32 张量}，给出时直接用作主权重（不再复制一份）并写回模型参数；
        不给时由模型参数转成 float32 得到。grad_dtype 是主权重梯度的精度（float32 或 bfloat16）。
        """
        self.grad_dtype = grad_dtype
        self.names = []
        self.params = []
        self.masters = []
        # 本来就是 float32 的参数，由优化器直接更新：[(名称, 参数)]
        self.direct = []
        self.handles = []
        if state is not None:
            check_state(named_parameters = named_parameters, state = state, what = "主权重")
        for name, parameter in named_parameters:
            if not parameter.requires_grad:
                raise ValueError(f"{name} 不需要梯度，不能建立主权重")
            if parameter.dtype == torch.float32:
                if state is not None:
                    with torch.no_grad():
                        parameter.copy_(state[name])
                self.direct.append((name, parameter))
                continue
            if state is not None:
                master = state[name].to(device = parameter.device, dtype = torch.float32)
            else:
                master = parameter.detach().to(dtype = torch.float32)
            master = torch.nn.Parameter(data = master, requires_grad = True)
            if grad_dtype != torch.float32:
                master.grad_dtype = grad_dtype
            self.names.append(name)
            self.params.append(parameter)
            self.masters.append(master)
            self.handles.append(parameter.register_post_accumulate_grad_hook(hook = self._hook(master = master, dtype = grad_dtype)))
        if state is not None:
            self.copy_to_model()

    @staticmethod
    def _hook(master, dtype):
        def accumulate(parameter):
            if master.grad is None:
                master.grad = parameter.grad.to(dtype = dtype)
            else:
                master.grad.add_(parameter.grad)
            parameter.grad = None
        return accumulate

    def optimizer_params(self):
        """交给优化器的张量：各主权重和本来就是 float32 的参数。"""
        return self.masters + [parameter for _, parameter in self.direct]

    def without_grad(self):
        """一次反传之后仍然没有梯度的参数名（梯度没有传到这些参数）。"""
        names = [name for name, master in zip(self.names, self.masters) if master.grad is None]
        return names + [name for name, parameter in self.direct if parameter.grad is None]

    def zero_model_grads(self):
        """清掉模型参数上残留的 bfloat16 梯度（正常情况下钩子已经清掉）。"""
        for parameter in self.params:
            parameter.grad = None

    @torch.no_grad()
    def copy_to_model(self):
        """把主权重舍入成模型参数的精度写回模型。"""
        for parameter, master in zip(self.params, self.masters):
            parameter.copy_(master)

    def state_dict(self):
        """续训用的全部主权重 {名称: float32 张量}，引用原张量，不复制。"""
        state = {name: master.detach() for name, master in zip(self.names, self.masters)}
        state.update({name: parameter.detach() for name, parameter in self.direct})
        return state

    def model_state(self, dtype = torch.bfloat16):
        """评测用的语言模型权重 {名称: CPU 上的 dtype 张量}，即模型中参与计算的权重。"""
        pairs = list(zip(self.names, self.params)) + self.direct
        return {
            name: parameter.detach().to(device = "cpu", dtype = dtype).contiguous()
            for name, parameter in pairs
        }

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


class DirectWeights:
    """没有主权重：优化器直接更新模型中的语言模型参数，接口与 MasterWeights 相同。"""

    # 续训状态 trainer_state.pt 中保存语言模型参数（bfloat16）的键。
    STATE_KEY = "language_model_weights"

    def __init__(self, named_parameters, state = None, grad_dtype = torch.bfloat16):
        """named_parameters 为 [(名称, 参数)]，都需要梯度。

        state 为续训时检查点中的 {名称: 张量}（保存时参数的精度），给出时写回模型参数。grad_dtype 为 float32 时，
        bfloat16 参数的梯度由 autograd 转成 float32 累加（参数的 grad_dtype）；为 bfloat16 时保持原样。
        """
        self.named = list(named_parameters)
        self.grad_dtype = grad_dtype
        if state is not None:
            check_state(named_parameters = self.named, state = state, what = "参数")
        for name, parameter in self.named:
            if not parameter.requires_grad:
                raise ValueError(f"{name} 不需要梯度，不能参与训练")
            if state is not None:
                if state[name].dtype != parameter.dtype:
                    raise ValueError(f"{name} 在检查点中为 {state[name].dtype}，模型中为 {parameter.dtype}")
                with torch.no_grad():
                    parameter.copy_(state[name])
            if parameter.dtype != torch.float32 and grad_dtype != parameter.dtype:
                parameter.grad_dtype = grad_dtype

    def optimizer_params(self):
        return [parameter for _, parameter in self.named]

    def without_grad(self):
        return [name for name, parameter in self.named if parameter.grad is None]

    def zero_model_grads(self):
        for _, parameter in self.named:
            parameter.grad = None

    def copy_to_model(self):
        """优化器已经直接更新了模型参数，不需要写回。"""

    def state_dict(self):
        """续训用的语言模型参数 {名称: 张量}，引用原张量，不复制。"""
        return {name: parameter.detach() for name, parameter in self.named}

    def model_state(self, dtype = torch.bfloat16):
        """评测用的语言模型权重 {名称: CPU 上的 dtype 张量}。"""
        return {
            name: parameter.detach().to(device = "cpu", dtype = dtype).contiguous()
            for name, parameter in self.named
        }

    def remove(self):
        """清掉梯度，恢复参数的 grad_dtype。"""
        for _, parameter in self.named:
            parameter.grad = None
            parameter.grad_dtype = parameter.dtype
