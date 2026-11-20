"""按参数组选择状态格式的 AdamW，计算用 torchao 0.18.0 的实现（任务 #66，选项见 precision.py）。

torchao 的 AdamW8bit、AdamW4bit、AdamWFp8 共用基类 torchao.optim.adam._AdamBase：逐个张量调用编译过的
single_param_adam，在 float32 中算出新的一阶矩、二阶矩和权重，再按各自的格式写回；bf16_stochastic_round 为真时，
bfloat16 参数的新权重用随机舍入写回（float32 参数不受影响）。这里沿用这个基类和它的计算，只改两处：

1. 状态的格式按参数组的 "states" 决定（_new_buffer）：fp32、bf16 是普通张量；8bit、4bit、fp8 用 torchao 对应
   优化器的量化张量和默认块大小，元素少于 4096 个或不能被块大小整除的张量与 torchao 一样按参数自身的精度存。
   这样 projector 组保持 float32 状态，语言模型组用选定的格式，两组在同一个优化器中，学习率调度和检查点格式不变。
2. 续训时保持状态的精度（load_state_dict）：torch 的 Optimizer.load_state_dict 会把浮点状态转成参数的精度，
   这里的状态可以与参数不同（例如 bfloat16 参数配 float32 状态），载入后按保存时的精度放回。

学习率在 torchao 中是参数组里的 float32 张量，学习率调度器就地改写它；scheduler.get_last_lr() 因此返回张量。
随机舍入的随机数来自编译后内核所在设备的默认生成器（CUDA 上即 torch.cuda 的默认生成器），训练程序保存和恢复它的
状态，续训与不中断的训练取到相同的随机数。
"""

import torch
from torchao.optim import AdamW4bit, AdamW8bit, AdamWFp8
from torchao.optim.adam import _AdamBase

# 量化状态：torchao 中对应的优化器类（取它的 _subclass_zeros）和它的默认块大小。
LOW_BIT = {
    "8bit": (AdamW8bit, 256),
    "4bit": (AdamW4bit, 128),
    "fp8": (AdamWFp8, 256),
}
PLAIN = {"fp32": torch.float32, "bf16": torch.bfloat16}
# torchao 只量化元素不少于这个数的张量（沿用 bitsandbytes 的做法）。
MIN_QUANTIZED = 4096


class GroupStateAdamW(_AdamBase):
    """AdamW（权重衰减与梯度分开），参数组的 "states" 取 fp32、bf16、8bit、4bit、fp8 之一，缺省为 fp32。"""

    def __init__(
        self,
        params,
        lr,
        betas,
        eps,
        weight_decay,
        bf16_stochastic_round = False,
    ):
        super().__init__(
            params = params,
            lr = lr,
            betas = betas,
            eps = eps,
            weight_decay = weight_decay,
            amsgrad = False,
            block_size = None,
            bf16_stochastic_round = bf16_stochastic_round,
            is_adamw = True,
        )

    def add_param_group(self, param_group):
        states = param_group.setdefault("states", "fp32")
        if states not in PLAIN and states not in LOW_BIT:
            raise ValueError(f"未知的状态格式 {states}")
        super().add_param_group(param_group = param_group)

    def _states_of(self, parameter):
        for group in self.param_groups:
            if any(item is parameter for item in group["params"]):
                return group["states"]
        raise KeyError("参数不在任何参数组中")

    def _new_buffer(self, p, signed):
        states = self._states_of(parameter = p)
        if states in PLAIN:
            return torch.zeros_like(input = p, dtype = PLAIN[states])
        optimizer_class, block_size = LOW_BIT[states]
        if p.numel() >= MIN_QUANTIZED and p.numel() % block_size == 0:
            return optimizer_class._subclass_zeros(p = p, signed = signed, block_size = block_size)
        return torch.zeros_like(input = p)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict = state_dict)
        saved_ids = [index for group in state_dict["param_groups"] for index in group["params"]]
        params = [parameter for group in self.param_groups for parameter in group["params"]]
        for index, parameter in zip(saved_ids, params):
            for key, value in state_dict["state"].get(index, {}).items():
                # 只处理普通张量；torchao 的量化状态以参数的精度作外观，载入时不被转换。
                if key == "step" or type(value) is not torch.Tensor or not value.is_floating_point():
                    continue
                if self.state[parameter][key].dtype != value.dtype:
                    self.state[parameter][key] = value.to(device = parameter.device)
