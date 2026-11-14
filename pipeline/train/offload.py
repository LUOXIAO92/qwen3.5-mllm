"""冻结层的梯度检查点、块流式加载和激活值卸载（实验 v2，任务 T59）。

做法照搬 moss-image-trainer（https://github.com/LUOXIAO92/moss-image-trainer）的 moss_lora/offload.py
和 trainer.py，按本项目的模型改写。三项都只作用于冻结的层（语言模型的解码层、音频塔的编码层，层数和维度都从模型读取，0.8B、2B、4B 通用），
训练的 projector 不受影响，同一批数据算出的损失和梯度与不用它们时在数值误差以内一致。

梯度检查点
==========

语言模型冻结，但梯度要穿过它传回 projector，所以前向时 autograd 会为每个解码层保存反传要用的中间结果
（投影、门控、归一化前后的值等），每个 token 约 2.2 MB，是训练显存的主要来源。打开梯度检查点后，
每个解码层前向时不保存中间结果，只留下这一层的输入（每个 token 每层 2 KB）；反传到这一层时用保存的输入
重算一遍这一层的前向，得到中间结果后立即求出输入的梯度，再交给前一层。参数冻结，重算只为求输入的梯度。
代价是语言模型多算一次前向。

块流式加载
==========

把一组层（语言模型或音频塔）末尾 count 个层的冻结权重放在锁页主机内存中，用到哪一层才拷到显存。拷贝在
独立的 CUDA 流上提前一层进行：计算流在算第 k 层时，拷贝流上传第 k + 1 层（反传重算时从后往前走，上传
第 k - 1 层），显存中同时最多有两层的权重（正在算的和正在上传的）。各层释放的显存缓冲按形状放进缓冲池，
交给下一个同形状的层复用，避免每步反复申请和释放整层大小的显存。权重冻结、不会改变，用完后直接把参数
指回主机上的副本，不像 moss 那样再拷回主机，拷贝量减半。

需要梯度的层（语言模型）流式加载时必须同时打开梯度检查点：否则 autograd 保存的是拷到显存的那份权重，
释放后反传会读到别的层的权重。音频塔不需要梯度，可以单独流式加载。

激活值卸载
==========

打开梯度检查点后，显存中还留着各解码层的输入。激活值卸载在前向时把 autograd 保存的张量（主要就是
这些层输入）在独立的拷贝流上搬到锁页主机内存，反传用到时再搬回来，搬运不阻塞 CPU 发射后面的内核。
只在前向期间生效；反传中重算的那一层的中间结果随用随放，不再搬运。
"""

import functools

import torch

# 同时驻留显存的流式层数：正在计算的一层和正在上传的一层。
WINDOW = 2
# 每种形状保留的空闲缓冲组数。遍历时有两组在用，第三组是刚释放、马上要交给下一层的。
POOL_SETS = 1


class BlockSwap:
    """把一组层末尾 count 个层的冻结参数放在锁页主机内存中，按需上传到显存。"""

    def __init__(self, blocks, count, device = "cuda"):
        self.blocks = blocks
        self.count = max(0, min(len(blocks), int(count)))
        self.start = len(blocks) - self.count
        self.device = torch.device(device)
        self.copy_stream = torch.cuda.Stream(device = self.device)
        # 层号 -> 冻结参数列表、对应的锁页主机副本
        self.params = {}
        self.host = {}
        # 层号 -> 当前绑定在参数上的显存张量、上传完成的事件
        self.resident = {}
        self.upload_done = {}
        # 形状签名 -> 释放出来的显存缓冲组
        self.pool = {}
        for index in range(self.start, len(blocks)):
            params = [parameter for parameter in blocks[index].parameters() if not parameter.requires_grad]
            host = [parameter.data.to(device = "cpu").pin_memory() for parameter in params]
            for parameter, buffer in zip(params, host):
                parameter.data = buffer
            self.params[index] = params
            self.host[index] = host
        torch.cuda.empty_cache()

    def offloaded_bytes(self):
        return sum(buffer.numel() * buffer.element_size() for host in self.host.values() for buffer in host)

    @staticmethod
    def _key(host):
        return tuple((tuple(buffer.shape), buffer.dtype) for buffer in host)

    def _upload(self, index):
        host = self.host[index]
        compute = torch.cuda.current_stream(device = self.device)
        bucket = self.pool.get(self._key(host = host))
        tensors = bucket.pop() if bucket else None
        if tensors is None:
            tensors = [torch.empty_like(input = buffer, device = self.device) for buffer in host]
        # 缓冲可能刚被计算流上排队中的内核读写过（缓冲池复用、或分配器回收的显存），拷贝前等计算流已排的工作。
        # 这一点在当前层的内核排队之前，所以拷贝仍与当前层的计算并行。
        self.copy_stream.wait_stream(stream = compute)
        with torch.cuda.stream(stream = self.copy_stream):
            for tensor, buffer in zip(tensors, host):
                tensor.copy_(other = buffer, non_blocking = True)
            uploaded = torch.cuda.Event()
            uploaded.record()
        # 缓冲由拷贝流写入；记录拷贝流，被缓冲池淘汰的缓冲在这次写入完成前不会被分配器回收。
        for tensor in tensors:
            tensor.record_stream(s = self.copy_stream)
        self.upload_done[index] = uploaded
        self.resident[index] = tensors
        for parameter, tensor in zip(self.params[index], tensors):
            parameter.data = tensor

    def wait(self, index):
        """让计算流等这一层上传完成。"""
        uploaded = self.upload_done.get(index)
        if index in self.resident and uploaded is not None:
            torch.cuda.current_stream(device = self.device).wait_event(event = uploaded)

    def prefetch(self, index):
        """提前上传一层，不等待。层不是流式的、已在显存中或两个位置都被占用时什么也不做。"""
        if index not in self.params or index in self.resident or len(self.resident) >= WINDOW:
            return
        self._upload(index = index)

    def load(self, index):
        if index not in self.params or index in self.resident:
            return
        if len(self.resident) >= WINDOW:
            # 正常遍历时提前上传总会空出一个位置；只有跳过某层时才会走到这里，淘汰离这一层最远的。
            self.release(index = max(self.resident, key = lambda other: abs(other - index)))
        self._upload(index = index)

    def release(self, index):
        """这一层的内核都已排队，把显存缓冲交还缓冲池，参数指回主机副本。

        之后复用这组缓冲的上传会先等计算流已排的工作（见 _upload），所以不会覆盖还没算完的权重。
        """
        tensors = self.resident.pop(index, None)
        if tensors is None:
            return
        host = self.host[index]
        bucket = self.pool.setdefault(self._key(host = host), [])
        if len(bucket) < POOL_SETS:
            bucket.append(tensors)
        for parameter, buffer in zip(self.params[index], host):
            parameter.data = buffer

    def restore(self):
        """把全部流式层的参数放回显存，取消流式加载。"""
        torch.cuda.synchronize(device = self.device)
        for index, params in self.params.items():
            for parameter, buffer in zip(params, self.host[index]):
                parameter.data = buffer.to(device = self.device)
        self.params, self.host, self.resident, self.upload_done, self.pool = {}, {}, {}, {}, {}
        self.count = 0
        self.start = len(self.blocks)


class _CheckpointBlock(torch.autograd.Function):
    """一层的梯度检查点：前向不保存中间结果，只保存输入；反传时（流式加载这一层后）重算并求输入的梯度。"""

    @staticmethod
    def forward(ctx, hidden_states, stack, index, forward, args, kwargs):
        ctx.save_for_backward(hidden_states)
        ctx.stack, ctx.index, ctx.forward, ctx.args, ctx.kwargs = stack, index, forward, args, kwargs
        stack.enter(index = index, direction = 1)
        output = forward(hidden_states, *args, **kwargs)
        stack.leave(index = index)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (hidden_states,) = ctx.saved_tensors
        stack, index = ctx.stack, ctx.index
        stack.enter(index = index, direction = -1)
        with torch.enable_grad():
            inputs = hidden_states.detach().requires_grad_(requires_grad = True)
            output = ctx.forward(inputs, *ctx.args, **ctx.kwargs)
        torch.autograd.backward(tensors = output, grad_tensors = grad_output)
        stack.leave(index = index)
        return inputs.grad, None, None, None, None, None


class FrozenBlocks:
    """接管一组冻结层（nn.ModuleList）的前向，按设置做梯度检查点和块流式加载。

    每层的 forward 换成一个包装：需要梯度时，末尾 checkpoint 个层走 _CheckpointBlock（checkpoint 为 True 时
    是全部层），其余层照常计算并保存中间结果；不需要梯度时（音频塔、dev 评估）直接计算。流式加载的层在计算前
    上传、计算后释放，并提前上传遍历方向上的下一层。需要梯度时流式加载的层必须在做检查点的层之内。
    """

    def __init__(
        self,
        blocks,
        checkpoint = False,
        swap_count = 0,
        device = "cuda",
    ):
        self.blocks = blocks
        if checkpoint is True:
            checkpoint = len(blocks)
        self.checkpoint = max(0, min(len(blocks), int(checkpoint or 0)))
        # 下标不小于 checkpoint_from 的层做梯度检查点。
        self.checkpoint_from = len(blocks) - self.checkpoint
        self.swap = None
        if swap_count > 0:
            self.swap = BlockSwap(blocks = blocks, count = swap_count, device = device)
        # 各层原来的前向；如果是实例上的属性（例如 torch.compile 编译后的前向），取消接管时放回去。
        self.originals = []
        for index, block in enumerate(blocks):
            original = block.forward
            self.originals.append((original, "forward" in block.__dict__))
            block.forward = functools.partial(self._call, index, original)

    def enter(self, index, direction):
        if self.swap is None:
            return
        self.swap.load(index = index)
        self.swap.wait(index = index)
        self.swap.prefetch(index = index + direction)

    def leave(self, index):
        if self.swap is not None:
            self.swap.release(index = index)

    def _call(self, index, original, hidden_states, *args, **kwargs):
        needs_grad = torch.is_grad_enabled() and hidden_states.requires_grad
        if needs_grad and index >= self.checkpoint_from:
            return _CheckpointBlock.apply(hidden_states, self, index, original, args, kwargs)
        if needs_grad and self.swap is not None and index in self.swap.params:
            raise RuntimeError("需要梯度的层流式加载时必须同时打开梯度检查点")
        self.enter(index = index, direction = 1)
        output = original(hidden_states, *args, **kwargs)
        self.leave(index = index)
        return output

    def offloaded_bytes(self):
        return 0 if self.swap is None else self.swap.offloaded_bytes()

    def remove(self):
        """恢复各层原来的前向，流式加载的权重放回显存。"""
        for block, (original, on_instance) in zip(self.blocks, self.originals):
            if on_instance:
                block.forward = original
            else:
                del block.forward
        if self.swap is not None:
            self.swap.restore()
            self.swap = None


class ActivationOffload(torch.autograd.graph.saved_tensors_hooks):
    """前向期间把 autograd 保存的显存张量搬到锁页主机内存，反传时搬回。

    与 torch.autograd.graph.save_on_cpu(pin_memory = True) 作用相同，区别是下行拷贝放在独立的流上、
    不阻塞 CPU：保存时拷贝流先等计算流算出这个张量，再异步拷到锁页内存，并记录完成事件；取回时计算流先等
    这个事件，再异步拷回显存。叶子张量（参数、批次输入）在别处还有引用，搬走也不省显存，留在原处；
    小于 min_bytes 的张量也留在显存。
    """

    def __init__(self, device = "cuda", min_bytes = 1 << 20):
        device = torch.device(device)
        copy_stream = torch.cuda.Stream(device = device)

        def pack(tensor):
            small = tensor.numel() * tensor.element_size() < min_bytes
            if tensor.device.type != "cuda" or tensor.is_leaf or small:
                return tensor
            host = torch.empty(
                size = tensor.size(),
                dtype = tensor.dtype,
                layout = tensor.layout,
                pin_memory = True,
            )
            copy_stream.wait_stream(stream = torch.cuda.current_stream(device = device))
            with torch.cuda.stream(stream = copy_stream):
                host.copy_(other = tensor, non_blocking = True)
                done = torch.cuda.Event()
                done.record()
            # 显存张量在拷贝完成前不能被分配器回收。
            tensor.record_stream(s = copy_stream)
            return (tensor.device, host, done)

        def unpack(packed):
            if isinstance(packed, torch.Tensor):
                return packed
            device_of, host, done = packed
            torch.cuda.current_stream(device = device_of).wait_event(event = done)
            return host.to(device = device_of, non_blocking = True)

        super().__init__(pack_hook = pack, unpack_hook = unpack)
