"""全量微调时语言模型参数组放在哪里更新（任务 #66）。

语言模型的参数在训练中有三种去处，同一个训练可以同时用到三种：

1. 显存（默认）：bfloat16 权重、主权重（或直接更新的参数）、梯度和 AdamW 状态都在显存中，由 master.py 的
   MasterWeights / DirectWeights 和 precision.make_optimizer 的优化器在 GPU 上更新，与以前完全相同。
2. 流式加载的层（--lm-blocks-to-swap N，offload.BlockSwap 的 trainable 模式）：语言模型末尾 N 层的 bfloat16 权重只在
   锁页主机内存中，前向和反传重算时提前一层上传、用完释放；反传算完一层后梯度搬到主机。主权重和 AdamW 状态也在
   主机内存中，由 HostAdamW 在 CPU 上更新主机内存中的权重，下一步上传的就是新权重。
3. 卸载优化器（--offload-optimizer）：其余（常驻显存的层、词嵌入兼输出层）元素不少于 HOST_MIN_NUMEL 个的语言模型参数，
   bfloat16 权重仍在显存中参与计算，梯度一算出就搬到锁页主机内存并释放显存中的梯度，主权重和 AdamW 状态在主机内存中，
   CPU 更新后把新的 bfloat16 权重拷回显存。搬运与更新用 torchao 0.18.0 的 CPUOffloadOptimizer（每个参数一份锁页的
   bfloat16 主机副本和梯度，反传中的钩子在独立的流上拷贝梯度，更新后再拷回），每个参数的 CPU 优化器是 HostAdamW。
   更小的张量（归一化层的权重等）留在第 1 种。

主机内存中的更新（HostAdamW）。梯度以 bfloat16 搬到主机，CPU 上转成 float32（精确），乘上裁剪系数，再做 AdamW：
- 有主权重（--language-model-master-weights fp32）：float32 主权重和两个 float32 状态在主机内存中，用 torch 的 CPU fused
  AdamW（torch._fused_adamw_，与显存中默认的 fused AdamW 同一种算法）按 4M 个元素一块更新，再就近舍入写回 bfloat16 权重。
- 没有主权重（none）：直接更新 bfloat16 权重，计算与 torchao 的 single_param_adam（显存中这类组合用的实现）相同，在 float32
  中算出新权重后就近或随机舍入（--language-model-rounding），用 torch.compile 编译成一个 CPU 内核。随机舍入的随机数来自
  HostContext 自己的生成器状态（CPU 上 torch.compile 的随机数取自默认生成器，更新时临时换上这份状态），随训练状态保存。
支持的精度组合：AdamW 状态必须是 fp32（低精度和量化状态是为省显存设计的，状态在主机内存时没有必要），并且 --grad-accum 1
（梯度每次反传后直接覆盖主机上的缓冲，不在主机上累加；torchao 的 offload_gradients 也有同样的限制）。梯度精度选项在这里
不起作用：--grad-accum 1 时 float32 梯度就是 bfloat16 梯度的精确转换。见 precision.host_problem。

裁剪梯度。显存中的梯度照常求范数；主机上的梯度在搬走之前已经在显存中求出各自的 float32 范数（offload.BlockSwap.
offload_grads、这里给卸载参数注册的钩子），一起算出总范数和裁剪系数：显存中的梯度原地乘系数，主机上的梯度在转成 float32
之后乘同一个系数（HostContext.scale）。只在显存中训练时仍直接调用 torch.nn.utils.clip_grad_norm_，与以前逐位相同。

续训状态（规范格式）。三种去处是同一套数学的不同存放位置，续训状态与只在显存中训练时的格式相同：优化器的 state_dict 只有
projector、语言模型两个参数组，语言模型参数按 MasterWeights / DirectWeights 交给优化器的顺序编号，每个参数的状态是
step、exp_avg、exp_avg_sq；trainer_state.pt 中的 language_model_master（或 language_model_weights）含全部语言模型参数。
所以这几个选项不是续训核对项：只在显存中训练的检查点可以打开它们续训，反过来也可以。CPU 与 GPU 上 AdamW 的舍入不同，
换位置续训时与不换位置的训练只在舍入误差上不同；同一种设置下停下再续训（确定性模式）逐位相同。
"""

import contextlib

import torch

from pipeline.train import precision
from pipeline.train.master import DirectWeights, MasterWeights, check_state

# 卸载优化器只接管元素不少于这个数的参数（torchao CPUOffloadOptimizer 的默认 minimal_size），更小的留在显存中。
HOST_MIN_NUMEL = 4096
# 主权重路径每块处理的元素个数：一块的 float32 梯度 16 MB，留在 CPU 缓存中。
CHUNK = 1 << 22
BETAS = (0.9, 0.999)
EPS = 1e-8


class HostContext:
    """同一个训练中各个 HostAdamW 共用：裁剪梯度的系数，随机舍入的随机数状态。"""

    def __init__(self, seed):
        self.scale = torch.ones(size = (), dtype = torch.float32)
        self.rng = torch.Generator().manual_seed(seed).get_state()

    @contextlib.contextmanager
    def own_rng(self):
        """临时把 CPU 默认生成器换成这份状态，结束时取回新状态、换回原来的状态。"""
        saved = torch.get_rng_state()
        torch.set_rng_state(new_state = self.rng)
        try:
            yield
        finally:
            self.rng = torch.get_rng_state()
            torch.set_rng_state(new_state = saved)


def _direct_update(
    p,
    grad,
    scale,
    step,
    exp_avg,
    exp_avg_sq,
    lr,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
    stochastic: bool,
):
    """torchao 0.18.0 的 single_param_adam（AdamW、无 amsgrad）的计算，多一步：梯度转成 float32 后乘裁剪系数 scale。"""
    p_f32 = p.float()
    grad_f32 = grad.float() * scale
    p_f32 = p_f32 - lr * weight_decay * p_f32
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step
    exp_avg_f32 = exp_avg.float().lerp(grad_f32, 1 - beta1)
    exp_avg_sq_f32 = exp_avg_sq.float().lerp(grad_f32.square(), 1 - beta2)
    exp_avg.copy_(exp_avg_f32)
    exp_avg_sq.copy_(exp_avg_sq_f32)
    denom = (exp_avg_sq_f32.sqrt() / bias_correction2.sqrt()) + eps
    p_f32 = p_f32 - lr * (exp_avg_f32 / bias_correction1) / denom
    if stochastic:
        p.copy_(_COMPILED["round"](p_f32))
    else:
        p.copy_(p_f32)


# 编译后的 _direct_update，以及它调用的 torchao 随机舍入函数（第一次用到时才导入 torchao）。
_COMPILED = {}


def compiled_direct_update():
    """编译成 CPU 内核的 _direct_update。张量都按一维传入、长度可变（dynamic），每种精度和舍入只编译一次。"""
    if "direct" not in _COMPILED:
        from torchao.optim.quant_utils import _fp32_to_bf16_sr

        _COMPILED["round"] = _fp32_to_bf16_sr
        _COMPILED["direct"] = torch.compile(
            model = _direct_update,
            fullgraph = True,
            dynamic = True,
        )
    return _COMPILED["direct"]


def adopt(tensor, dtype = torch.float32):
    """续训或调整位置时接收一份主机上的状态：已经是 CPU 上连续的同精度张量时直接使用，否则转换一份。"""
    if tensor.device.type == "cpu" and tensor.dtype == dtype and tensor.is_contiguous():
        return tensor
    return tensor.to(device = "cpu", dtype = dtype).contiguous()


class HostAdamW(torch.optim.Optimizer):
    """在 CPU 上更新主机内存中的语言模型参数的 AdamW（权重衰减与梯度分开）。

    参数是主机内存中的权重（流式加载层的锁页权重，或 torchao 的锁页主机副本），.grad 是从显存搬来的同形状梯度。
    master 为真时每个 bfloat16 参数另有 float32 主权重（self.masters，第一次更新时由权重转换得到，续训时由 set_master
    给出），更新主权重后就近舍入写回权重；为假时直接更新权重，stochastic 决定 bfloat16 的舍入。状态 step、exp_avg、
    exp_avg_sq 与 torch 的 AdamW 相同，都是 CPU 上的 float32 张量。context 是共用的 HostContext。
    """

    def __init__(
        self,
        params,
        lr,
        betas,
        eps,
        weight_decay,
        master,
        stochastic,
        context,
    ):
        super().__init__(params = params, defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})
        self.master = master
        self.stochastic = stochastic
        self.context = context
        self.masters = {}
        self._chunk = None

    def master_of(self, param):
        """param 的 float32 主权重（没有主权重的设置或 float32 参数返回参数本身）。"""
        if not self.master or param.dtype == torch.float32:
            return param
        if param not in self.masters:
            self.masters[param] = param.detach().to(dtype = torch.float32)
        return self.masters[param]

    def set_master(self, param, value):
        self.masters[param] = adopt(tensor = value)

    def load_entry(self, param, entry):
        """载入规范格式中一个参数的 AdamW 状态（step、exp_avg、exp_avg_sq）。"""
        for key in ("exp_avg", "exp_avg_sq"):
            if tuple(entry[key].shape) != tuple(param.shape):
                raise ValueError(f"{key} 形状为 {tuple(entry[key].shape)}，参数为 {tuple(param.shape)}")
        self.state[param] = {
            "step": entry["step"].detach().to(device = "cpu", dtype = torch.float32).clone(),
            "exp_avg": adopt(tensor = entry["exp_avg"]),
            "exp_avg_sq": adopt(tensor = entry["exp_avg_sq"]),
        }

    def _chunk_buffer(self):
        if self._chunk is None:
            self._chunk = torch.empty(size = (CHUNK,), dtype = torch.float32)
        return self._chunk

    def _update_master(self, param, state, group):
        """torch 的 CPU fused AdamW 更新主权重（float32 参数就是它自己），按块进行，再写回 bfloat16 权重。"""
        target = self.master_of(param = param)
        weight = None if target is param else param.view(-1)
        target_flat = target.view(-1)
        grad = param.grad.view(-1)
        exp_avg = state["exp_avg"].view(-1)
        exp_avg_sq = state["exp_avg_sq"].view(-1)
        buffer = self._chunk_buffer()
        beta1, beta2 = group["betas"]
        lr = float(group["lr"])
        for start in range(0, target_flat.numel(), CHUNK):
            end = min(start + CHUNK, target_flat.numel())
            chunk = buffer[: end - start]
            chunk.copy_(grad[start:end])
            chunk.mul_(self.context.scale)
            torch._fused_adamw_(
                [target_flat[start:end]],
                [chunk],
                [exp_avg[start:end]],
                [exp_avg_sq[start:end]],
                [],
                [state["step"]],
                lr = lr,
                beta1 = beta1,
                beta2 = beta2,
                weight_decay = group["weight_decay"],
                eps = group["eps"],
                amsgrad = False,
                maximize = False,
            )
            if weight is not None:
                weight[start:end].copy_(target_flat[start:end])

    def _update_direct(self, param, state, group):
        beta1, beta2 = group["betas"]
        stochastic = self.stochastic and param.dtype == torch.bfloat16
        context = self.context.own_rng() if stochastic else contextlib.nullcontext()
        with context:
            compiled_direct_update()(
                param.view(-1),
                param.grad.view(-1),
                self.context.scale,
                state["step"],
                state["exp_avg"].view(-1),
                state["exp_avg_sq"].view(-1),
                torch.tensor(data = float(group["lr"]), dtype = torch.float32),
                beta1,
                beta2,
                group["weight_decay"],
                group["eps"],
                stochastic,
            )

    @torch.no_grad()
    def step(self, closure = None):
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if not state:
                    state["step"] = torch.zeros(size = (), dtype = torch.float32)
                    state["exp_avg"] = torch.zeros_like(input = param, dtype = torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(input = param, dtype = torch.float32)
                state["step"] += 1
                if self.master:
                    self._update_master(param = param, state = state, group = group)
                else:
                    self._update_direct(param = param, state = state, group = group)
        return None


class UpdateOptimizer(torch.optim.Optimizer):
    """交给学习率调度器和训练循环的优化器：参数组就是显存中优化器的两组（projector、语言模型），学习率调度器改写它们，
    更新时把语言模型组的学习率同步给主机上的优化器；state_dict 是规范格式（见模块说明）。

    与 torchao 的 CPUOffloadOptimizer 一样不调用 Optimizer.__init__，只为让学习率调度器接受它。
    """

    def __init__(self, update):
        self.update = update

    @property
    def param_groups(self):
        return self.update.gpu.param_groups

    def step(self, closure = None):
        self.update.step()

    def zero_grad(self, set_to_none = True):
        self.update.zero_grad()

    def state_dict(self):
        return self.update.state_dict()

    def load_state_dict(self, state_dict):
        self.update.load_state_dict(state_dict = state_dict)


class LanguageModelUpdate:
    """全量微调时语言模型参数组的权重、优化器与它们的存放位置；接口与 master.py 的两个类相同，另有 optimizer、
    clip_grad_norm 和 rebuild。

    named_parameters 为全部语言模型参数 [(名称, 参数)]；applied 是 optimize.Applied（流式加载的层由它决定）；
    state 为续训时 trainer_state.pt 中 STATE_KEY 下的 {名称: 张量}（规范格式，全部语言模型参数）。
    """

    def __init__(
        self,
        named_parameters,
        projector_params,
        options,
        lr,
        language_model_lr,
        weight_decay,
        applied,
        offload_optimizer = False,
        seed = 0,
        state = None,
    ):
        self.named = list(named_parameters)
        self.projector_params = list(projector_params)
        self.options = dict(options)
        self.master = options["language_model_master_weights"] == "fp32"
        self.stochastic = options["language_model_rounding"] == "stochastic"
        self.grad_dtype = precision.DTYPES[options["language_model_grad_dtype"]]
        self.lr = lr
        self.language_model_lr = language_model_lr
        self.weight_decay = weight_decay
        self.applied = applied
        self.offload_optimizer = offload_optimizer
        weights_class = MasterWeights if self.master else DirectWeights
        self.weights_class = weights_class
        self.STATE_KEY = weights_class.STATE_KEY
        # 规范顺序：只在显存中训练时语言模型参数交给优化器的顺序（MasterWeights 先 bfloat16 参数、后 float32 参数）。
        if self.master:
            self.canonical = [name for name, parameter in self.named if parameter.dtype != torch.float32]
            self.canonical += [name for name, parameter in self.named if parameter.dtype == torch.float32]
        else:
            self.canonical = [name for name, _ in self.named]
        self.context = HostContext(seed = seed)
        self.optimizer = UpdateOptimizer(update = self)
        if state is not None:
            check_state(named_parameters = self.named, state = state, what = "权重")
        self._build(state = state)

    # 组建与拆除

    def _build(self, state):
        torch.cuda.synchronize()
        lm = self.applied.lm if self.applied is not None else None
        streamed = {id(parameter): (host, grad) for parameter, host, grad in (lm.streamed_parameters() if lm is not None else [])}
        resident = [(name, parameter) for name, parameter in self.named if id(parameter) not in streamed]
        offloaded = [
            (name, parameter) for name, parameter in resident
            if self.offload_optimizer and parameter.numel() >= HOST_MIN_NUMEL
        ]
        offloaded_ids = {id(parameter) for _, parameter in offloaded}
        gpu = [(name, parameter) for name, parameter in resident if id(parameter) not in offloaded_ids]
        self.gpu_named = gpu
        self.offloaded_named = offloaded
        self.streamed_named = [(name, parameter) for name, parameter in self.named if id(parameter) in streamed]
        self.weights = self.weights_class(
            named_parameters = gpu,
            state = None if state is None else {name: state[name] for name, _ in gpu},
            grad_dtype = self.grad_dtype,
        )
        self.gpu = precision.make_optimizer(
            projector_params = self.projector_params,
            language_model_params = self.weights.optimizer_params(),
            lr = self.lr,
            language_model_lr = self.language_model_lr,
            weight_decay = self.weight_decay,
            options = self.options,
        )
        # 名称 -> (HostAdamW, 它更新的主机张量)
        self.host_entries = {}
        host_kwargs = {
            "lr": self.language_model_lr,
            "betas": BETAS,
            "eps": EPS,
            "weight_decay": self.weight_decay,
            "master": self.master,
            "stochastic": self.stochastic,
            "context": self.context,
        }
        self.offload = None
        self.offload_norms = {}
        self.hook_handles = []
        self.torchao_hooks = []
        if offloaded:
            from torchao.optim import CPUOffloadOptimizer

            if state is not None:
                with torch.no_grad():
                    for name, parameter in offloaded:
                        parameter.copy_(state[name])
            for _, parameter in offloaded:
                # 先于 torchao 的钩子注册：它把梯度拷走后就释放显存中的梯度。
                self.hook_handles.append(parameter.register_post_accumulate_grad_hook(hook = self._record_norm))
            before = {id(parameter): set(parameter._post_accumulate_grad_hooks or {}) for _, parameter in offloaded}
            self.offload = CPUOffloadOptimizer(
                [parameter for _, parameter in offloaded],
                optimizer_class = HostAdamW,
                offload_gradients = True,
                minimal_size = 0,
                **host_kwargs,
            )
            for _, parameter in offloaded:
                added = set(parameter._post_accumulate_grad_hooks or {}) - before[id(parameter)]
                self.torchao_hooks.append((parameter, added))
            # torchao 在当前流上异步拷出主机副本，读它们之前等拷完。
            torch.cuda.synchronize()
            for name, parameter in offloaded:
                self.host_entries[name] = (self.offload.optim_dict[parameter], self.offload.param_d2h_map[parameter])
        self.swap = None
        self.stream = None
        if self.streamed_named:
            self.swap = lm.swap
            hosts = []
            for name, parameter in self.streamed_named:
                host, grad = streamed[id(parameter)]
                host.grad = grad
                if state is not None:
                    host.copy_(state[name])
                hosts.append(host)
            self.stream = HostAdamW(params = hosts, **host_kwargs)
            for (name, _), host in zip(self.streamed_named, hosts):
                self.host_entries[name] = (self.stream, host)
        if state is not None and self.master:
            for name, (optimizer, host) in self.host_entries.items():
                if host.dtype != torch.float32:
                    optimizer.set_master(param = host, value = state[name])

    def _record_norm(self, parameter):
        self.offload_norms[id(parameter)] = torch.linalg.vector_norm(input = parameter.grad, dtype = torch.float32)

    def _teardown(self):
        torch.cuda.synchronize()
        self.weights.remove()
        for handle in self.hook_handles:
            handle.remove()
        for parameter, keys in self.torchao_hooks:
            for key in keys:
                parameter._post_accumulate_grad_hooks.pop(key, None)
            parameter.grad = None
        if self.swap is not None:
            for _, host in self.host_entries.values():
                host.grad = None
        self.hook_handles, self.torchao_hooks = [], []
        self.offload = self.stream = self.swap = self.gpu = None
        self.host_entries = {}

    def rebuild(self):
        """流式加载的层改变以后（显存不足时的退让）按新的位置重新安排，权重和优化器状态原样搬过去。"""
        weights = self.weights_state()
        optimizer_state = self.state_dict()
        self._teardown()
        self._build(state = weights)
        self.load_state_dict(state_dict = optimizer_state)
        del weights, optimizer_state
        torch.cuda.empty_cache()

    # 训练循环用到的接口

    def _gpu_names(self):
        """显存中优化器的语言模型组里参数的名称，按交给优化器的顺序。"""
        if self.master:
            return self.weights.names + [name for name, _ in self.weights.direct]
        return [name for name, _ in self.weights.named]

    def step(self):
        lr = float(self.gpu.param_groups[1]["lr"])
        self.gpu.step()
        if self.offload is not None:
            for optimizer in self.offload.optim_dict.values():
                optimizer.param_groups[0]["lr"] = lr
            self.offload.step()
        if self.stream is not None:
            self.swap.wait_grads()
            self.stream.param_groups[0]["lr"] = lr
            self.stream.step()

    def zero_grad(self):
        self.gpu.zero_grad(set_to_none = True)
        if self.offload is not None:
            self.offload.zero_grad()
            self.offload.queue.clear()
            self.offload_norms = {}
        if self.swap is not None:
            self.swap.clear_grads()

    def zero_model_grads(self):
        self.weights.zero_model_grads()
        for _, parameter in self.offloaded_named + self.streamed_named:
            parameter.grad = None
        self.zero_grad()

    def without_grad(self):
        missing = self.weights.without_grad()
        if self.offload is not None:
            missing += [name for name, parameter in self.offloaded_named if parameter not in self.offload.queue]
        if self.swap is not None:
            missing += [name for name, parameter in self.streamed_named if id(parameter) not in self.swap.received]
        return missing

    def clip_grad_norm(self, max_norm):
        """裁剪全部可训练参数（projector 与语言模型）的梯度，返回裁剪前的总范数（显存中的 0 维张量）。"""
        tensors = self.projector_params + self.weights.optimizer_params()
        if not self.host_entries:
            return torch.nn.utils.clip_grad_norm_(parameters = tensors, max_norm = max_norm)
        grads = [tensor.grad for tensor in tensors if tensor.grad is not None]
        norms = [torch.nn.utils.get_total_norm(tensors = grads)] if grads else []
        norms += list(self.offload_norms.values())
        if self.swap is not None:
            norms += [norm for layer in self.swap.grad_norms.values() for norm in layer]
        total = torch.linalg.vector_norm(input = torch.stack(tensors = [norm.to(device = "cuda") for norm in norms]))
        torch.nn.utils.clip_grads_with_norm_(parameters = tensors, max_norm = max_norm, total_norm = total)
        # 与 clip_grads_with_norm_ 相同的系数，用于主机上的梯度（转成 float32 之后再乘）。
        coefficient = torch.clamp(input = max_norm / (total + 1e-6), max = 1.0)
        self.context.scale.copy_(coefficient)
        return total

    def copy_to_model(self):
        self.weights.copy_to_model()

    def weights_state(self):
        """续训用的全部语言模型权重 {名称: 张量}（规范格式）：主权重，或没有主权重时优化器直接更新的参数。"""
        state = self.weights.state_dict()
        for name, (optimizer, host) in self.host_entries.items():
            state[name] = optimizer.master_of(param = host).detach()
        return state

    def state_dict(self):
        """规范格式的优化器状态（见模块说明）。"""
        saved = self.gpu.state_dict()
        if not self.host_entries:
            return saved
        base = len(self.projector_params)
        local = self._gpu_names()
        canonical_index = {name: base + position for position, name in enumerate(self.canonical)}
        state = {}
        for index, entry in saved["state"].items():
            state[index if index < base else canonical_index[local[index - base]]] = entry
        for name, (optimizer, host) in self.host_entries.items():
            entry = optimizer.state.get(host)
            if entry:
                state[canonical_index[name]] = {key: entry[key] for key in ("step", "exp_avg", "exp_avg_sq")}
        groups = [dict(group) for group in saved["param_groups"]]
        groups[1]["params"] = list(range(base, base + len(self.canonical)))
        return {"state": dict(sorted(state.items())), "param_groups": groups}

    def load_state_dict(self, state_dict):
        if not self.host_entries:
            self.gpu.load_state_dict(state_dict = state_dict)
            return
        base = len(self.projector_params)
        local = self._gpu_names()
        local_index = {name: base + position for position, name in enumerate(local)}
        state = {}
        for index, entry in state_dict["state"].items():
            if index < base:
                state[index] = entry
                continue
            name = self.canonical[index - base]
            if name in local_index:
                state[local_index[name]] = entry
            else:
                optimizer, host = self.host_entries[name]
                optimizer.load_entry(param = host, entry = entry)
        groups = [dict(group) for group in state_dict["param_groups"]]
        if len(groups[1]["params"]) != len(self.canonical):
            raise ValueError(f"优化器状态中语言模型组有 {len(groups[1]['params'])} 个参数，模型中有 {len(self.canonical)} 个")
        groups[0]["params"] = list(range(base))
        groups[1]["params"] = list(range(base, base + len(local)))
        self.gpu.load_state_dict(state_dict = {"state": state, "param_groups": groups})

    def model_state(self, dtype = torch.bfloat16):
        """评测用的语言模型权重 {名称: CPU 上的 dtype 张量}，即模型中参与计算的权重（流式加载的层在主机内存中）。"""
        return {
            name: parameter.detach().to(device = "cpu", dtype = dtype).contiguous()
            for name, parameter in self.named
        }

    def remove(self):
        self._teardown()

    def describe(self):
        """训练开始时打印的概况：三种去处各有多少参数，主机内存中的主权重与状态估计（GB，10^9 字节）。"""
        count = lambda pairs: sum(parameter.numel() for _, parameter in pairs)
        host = count(pairs = self.offloaded_named) + count(pairs = self.streamed_named)
        per_parameter = (4 if self.master else 0) + 8
        return {
            "gpu_parameters": count(pairs = self.gpu_named),
            "offloaded_parameters": count(pairs = self.offloaded_named),
            "streamed_parameters": count(pairs = self.streamed_named),
            # 主机内存中的主权重和 AdamW 状态（不含锁页的 bfloat16 权重与梯度缓冲）。
            "host_master_and_states_gb": round(number = host * per_parameter / 1e9, ndigits = 2),
            # 锁页内存：流式层的权重与梯度缓冲、torchao 的主机副本与梯度，各 2 字节（锁页分配器另有按 2 的幂取整的余量）。
            "host_pinned_gb": round(number = host * 4 / 1e9, ndigits = 2),
        }
