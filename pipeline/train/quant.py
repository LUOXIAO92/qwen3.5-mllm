"""冻结权重的 fp8 存储（实验 v2，任务 T59，默认关闭）。

照搬 moss-image-trainer 的 moss_lora/quant.py 中的 FP8Linear：冻结的 nn.Linear 权重按输出行各取一个缩放系数，
除以缩放后存成 fp8（e4m3，每个值 1 字节），每次前向再乘回缩放、转成 bfloat16 计算。参数只读不训练，
梯度照常经过输入传回去。语言模型解码层的权重从约 1.0 GB 降到 0.5 GB，音频塔编码层从约 1.26 GB 降到
0.63 GB；与块流式加载一起用时，每步要拷贝的字节数也减半。代价是权重带上 fp8 的舍入误差（相对误差约 2⁻⁴），
损失与 bfloat16 不再在数值误差以内一致，所以默认关闭，只在显存确实不够时使用。
"""

import torch
from torch import nn

FP8_MAX = 448.0


class FP8Linear(nn.Linear):
    """冻结的线性层，权重存为 fp8 e4m3，每个输出行一个缩放系数，前向时还原成计算精度。

    继承 nn.Linear 只为保留类型判断；构造时在 meta 设备上初始化，不额外占内存，真正的权重随后替换进来。
    权重和缩放都登记为参数（不需要梯度），块流式加载按参数搬运，两者都会随层一起进出显存。
    """

    def __init__(self, linear, compute_dtype = torch.bfloat16):
        super().__init__(
            in_features = linear.in_features,
            out_features = linear.out_features,
            bias = linear.bias is not None,
            device = "meta",
            dtype = compute_dtype,
        )
        self.compute_dtype = compute_dtype
        weight = linear.weight.detach().float()
        scale = weight.abs().amax(dim = 1, keepdim = True).clamp_min(min = 1e-8) / FP8_MAX
        self.weight = nn.Parameter(data = (weight / scale).to(dtype = torch.float8_e4m3fn), requires_grad = False)
        self.scale = nn.Parameter(data = scale.to(dtype = compute_dtype), requires_grad = False)
        self.bias = linear.bias
        if self.bias is not None:
            self.bias.requires_grad_(requires_grad = False)

    def forward(self, input):
        dtype = input.dtype if input.dtype in (torch.bfloat16, torch.float16, torch.float32) else self.compute_dtype
        weight = self.weight.to(dtype = dtype) * self.scale.to(dtype = dtype)
        bias = None if self.bias is None else self.bias.to(dtype = dtype)
        return nn.functional.linear(input = input.to(dtype = dtype), weight = weight, bias = bias)


def quantize_fp8(module, compute_dtype = torch.bfloat16):
    """把 module 中全部冻结的 nn.Linear 换成 FP8Linear，返回替换的个数。需要梯度的线性层不动。"""
    targets = [
        (name, child) for name, child in module.named_modules()
        if isinstance(child, nn.Linear) and not isinstance(child, FP8Linear) and not child.weight.requires_grad
    ]
    for name, child in targets:
        parent_name, _, leaf = name.rpartition(".")
        parent = module.get_submodule(target = parent_name) if parent_name else module
        setattr(parent, leaf, FP8Linear(linear = child, compute_dtype = compute_dtype))
    return len(targets)
