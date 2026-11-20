"""梯度检查点、块流式加载和激活值卸载（实验 v2，任务 T59；可训练层的流式加载与层边界的提前上传，任务 #66）。

做法照搬 moss-image-trainer（https://github.com/LUOXIAO92/moss-image-trainer）的 moss_lora/offload.py
和 trainer.py，按本项目的模型改写。三项作用于语言模型的解码层和音频塔的编码层（层数和维度都从模型读取，0.8B、2B、4B
通用），只改变权重和激活值放在哪里，同一批数据算出的损失和梯度与不用它们时一致。

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

需要更新的层（全量微调，任务 #66）也可以流式加载（BlockSwap 的 trainable 模式）：锁页主机内存中的 bfloat16 权重
是唯一的一份，前向和反传重算时按上面的做法提前一层上传，用完释放；反传算完一层后，这一层各参数的梯度先在显存中
求出各自的 float32 范数（裁剪梯度用），再在拷贝流上搬到锁页主机内存中的梯度缓冲，显存中的梯度随即释放。主机上的
AdamW（placement.py）在一步结束时更新主机内存中的权重，下一步上传的就是更新后的权重。

激活值卸载
==========

打开梯度检查点后，显存中还留着各解码层的输入（层边界）。激活值卸载在前向时把它们在独立的拷贝流上搬到锁页主机内存，
反传时搬回，搬运不阻塞 CPU 发射后面的内核。层边界的取回与流式加载用同一种提前一层的做法（BoundaryOffload）：
反传算第 i 层时，拷贝流已经在上传第 i - 1 层的边界，两者用事件同步；每层的锁页主机缓冲和显存缓冲在各步之间复用。
其余 autograd 保存的张量（不做检查点的层的中间结果等）仍由 ActivationOffload 的钩子在保存时搬走、用到时搬回，不提前。
只在前向期间生效；反传中重算的那一层的中间结果随用随放，不再搬运。
"""

import functools

import torch

# 同时驻留显存的流式层数：正在计算的一层和正在上传的一层。
WINDOW = 2
# 每种形状保留的空闲缓冲组数。遍历时有两组在用，第三组是刚释放、马上要交给下一层的。
POOL_SETS = 1


class BlockSwap:
    """把一组层末尾 count 个层的参数放在锁页主机内存中，按需上传到显存。

    trainable 为假时只搬冻结的参数；为真时（全量微调的语言模型）搬这些层的全部参数，需要梯度的参数另有一份锁页的
    梯度缓冲：反传算完一层后调用 offload_grads，把梯度搬到主机并释放显存中的梯度。
    """

    def __init__(
        self,
        blocks,
        count,
        device = "cuda",
        trainable = False,
    ):
        self.blocks = blocks
        self.count = max(0, min(len(blocks), int(count)))
        self.start = len(blocks) - self.count
        self.device = torch.device(device)
        self.trainable = trainable
        self.copy_stream = torch.cuda.Stream(device = self.device)
        # 层号 -> 流式加载的参数列表、对应的锁页主机副本
        self.params = {}
        self.host = {}
        # 层号 -> 各参数的锁页梯度缓冲（不需要梯度的参数为 None）、梯度搬完的事件、各梯度的 float32 范数（显存中的 0 维张量）
        self.host_grads = {}
        self.grad_done = {}
        self.grad_norms = {}
        # 本步梯度已搬到主机的参数（id）
        self.received = set()
        # 层号 -> 当前绑定在参数上的显存张量、上传完成的事件
        self.resident = {}
        self.upload_done = {}
        # 形状签名 -> 释放出来的显存缓冲组
        self.pool = {}
        for index in range(self.start, len(blocks)):
            params = [parameter for parameter in blocks[index].parameters() if trainable or not parameter.requires_grad]
            host = [parameter.data.to(device = "cpu").pin_memory() for parameter in params]
            for parameter, buffer in zip(params, host):
                parameter.data = buffer
            self.params[index] = params
            self.host[index] = host
            if trainable:
                self.host_grads[index] = [
                    torch.empty_like(input = buffer, pin_memory = True) if parameter.requires_grad else None
                    for parameter, buffer in zip(params, host)
                ]
        torch.cuda.empty_cache()

    def offloaded_bytes(self):
        """放在主机内存中的权重和梯度缓冲的字节数。"""
        buffers = [buffer for host in self.host.values() for buffer in host]
        buffers += [buffer for grads in self.host_grads.values() for buffer in grads if buffer is not None]
        return sum(buffer.numel() * buffer.element_size() for buffer in buffers)

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

    def offload_grads(self, index):
        """这一层的反传已全部排队：求出各梯度的 float32 范数，把梯度搬到锁页主机内存，释放显存中的梯度。

        拷贝流先等计算流算出梯度；显存中的梯度记录拷贝流，在搬完之前不会被分配器回收。梯度累积不在这里处理：
        每次反传都覆盖主机上的梯度缓冲（placement.py 要求 --grad-accum 1）。
        """
        compute = torch.cuda.current_stream(device = self.device)
        pairs = [
            (parameter, buffer) for parameter, buffer in zip(self.params[index], self.host_grads[index])
            if buffer is not None and parameter.grad is not None
        ]
        self.grad_norms[index] = [
            torch.linalg.vector_norm(input = parameter.grad, dtype = torch.float32) for parameter, _ in pairs
        ]
        self.copy_stream.wait_stream(stream = compute)
        with torch.cuda.stream(stream = self.copy_stream):
            for parameter, buffer in pairs:
                buffer.copy_(other = parameter.grad, non_blocking = True)
            done = torch.cuda.Event()
            done.record()
        for parameter, _ in pairs:
            parameter.grad.record_stream(s = self.copy_stream)
            parameter.grad = None
        self.grad_done[index] = done
        self.received.update(id(parameter) for parameter, _ in pairs)

    def wait_grads(self):
        """让 CPU 等各层的梯度搬完（主机上的优化器读梯度之前调用）。"""
        for done in self.grad_done.values():
            done.synchronize()

    def clear_grads(self):
        """一步结束或显存不足重试前清掉本步的梯度记录（主机缓冲下一次反传时覆盖）。"""
        for params in self.params.values():
            for parameter in params:
                parameter.grad = None
        self.grad_done, self.grad_norms, self.received = {}, {}, set()

    def restore(self):
        """把全部流式层的参数放回显存，取消流式加载。可训练的层以主机上的权重（优化器更新的那一份）为准。"""
        torch.cuda.synchronize(device = self.device)
        for index, params in self.params.items():
            for parameter, buffer in zip(params, self.host[index]):
                parameter.data = buffer.to(device = self.device)
        self.params, self.host, self.resident, self.upload_done, self.pool = {}, {}, {}, {}, {}
        self.host_grads, self.grad_done, self.grad_norms, self.received = {}, {}, {}, set()
        self.count = 0
        self.start = len(self.blocks)


class Boundary:
    """一个层边界的去向：主机缓冲中的副本、拷完的事件，以及提前上传后的显存张量和上传完的事件。"""

    def __init__(self, store, index, host, saved):
        self.store = store
        self.index = index
        self.host = host
        self.saved = saved
        self.device_tensor = None
        self.uploaded = None


class BoundaryOffload:
    """梯度检查点各层的输入（层边界）：前向时拷到锁页主机内存，反传时提前一层拷回显存。

    做法与 BlockSwap 相同：拷贝在独立的流上进行，用事件同步。反传算第 i 层时先取回第 i 层的边界（通常已在算第 i + 1 层
    时上传好），再让拷贝流上传第 i - 1 层的边界，与第 i 层的重算和反传同时进行。每层一块锁页主机缓冲，不够大时扩大，
    在各步之间复用；上传用的显存缓冲用完放回缓冲池（只留当前形状的一组），交给下一次上传。prefetch 为假时不提前上传，
    用到时才上传并等待（用来测量提前上传省下的时间）。
    """

    # 主机缓冲扩大时多留的比例，免得批次稍长一点就重新申请锁页内存。
    GROWTH = 1.25

    def __init__(self, device = "cuda", prefetch = True):
        self.device = torch.device(device)
        self.prefetch_enabled = prefetch
        self.copy_stream = torch.cuda.Stream(device = self.device)
        # 层号 -> 锁页主机缓冲（一维）；层号 -> 本次前向保存的 Boundary（提前上传时按层号找下一个）
        self.host = {}
        self.saved = {}
        # (形状, 精度) -> 用完的显存缓冲
        self.pool = {}

    def host_bytes(self):
        return sum(buffer.numel() * buffer.element_size() for buffer in self.host.values())

    def save(self, index, tensor):
        """前向：在拷贝流上把 tensor 拷到第 index 层的主机缓冲，返回 Boundary。显存中的张量随调用者释放。"""
        numel = tensor.numel()
        buffer = self.host.get(index)
        if buffer is None or buffer.dtype != tensor.dtype or buffer.numel() < numel:
            # 旧缓冲上可能还有排队的拷贝；锁页内存分配器记录了这些拷贝的事件，完成前不会把它交给别处。
            buffer = torch.empty(
                size = (int(numel * self.GROWTH),),
                dtype = tensor.dtype,
                pin_memory = True,
            )
            self.host[index] = buffer
        host = buffer[:numel].view(tensor.shape)
        # 上一步反传从这块缓冲上传的拷贝也在拷贝流上、排在前面，覆盖它不会读写冲突。
        self.copy_stream.wait_stream(stream = torch.cuda.current_stream(device = self.device))
        with torch.cuda.stream(stream = self.copy_stream):
            host.copy_(other = tensor, non_blocking = True)
            saved = torch.cuda.Event()
            saved.record()
        # 显存中的张量在拷完之前不能被分配器回收。
        tensor.record_stream(s = self.copy_stream)
        boundary = Boundary(
            store = self,
            index = index,
            host = host,
            saved = saved,
        )
        self.saved[index] = boundary
        return boundary

    def _upload(self, boundary):
        key = (tuple(boundary.host.shape), boundary.host.dtype)
        bucket = self.pool.get(key)
        tensor = bucket.pop() if bucket else None
        if tensor is None:
            tensor = torch.empty(size = boundary.host.shape, dtype = boundary.host.dtype, device = self.device)
            # 显存缓冲由拷贝流写入；记录拷贝流，被淘汰时在写入完成前不会被分配器回收。
            tensor.record_stream(s = self.copy_stream)
        # 缓冲池中的缓冲可能还在被计算流上排队的内核读取，上传前等计算流已排的工作。这一点在当前层的内核排队之前，
        # 所以上传仍与当前层的计算并行。
        self.copy_stream.wait_stream(stream = torch.cuda.current_stream(device = self.device))
        with torch.cuda.stream(stream = self.copy_stream):
            tensor.copy_(other = boundary.host, non_blocking = True)
            uploaded = torch.cuda.Event()
            uploaded.record()
        boundary.device_tensor = tensor
        boundary.uploaded = uploaded

    def prefetch(self, index):
        """提前上传第 index 层的边界，不等待；没有这一层、已经上传或不提前上传时什么也不做。"""
        boundary = self.saved.get(index)
        if not self.prefetch_enabled or boundary is None or boundary.device_tensor is not None:
            return
        self._upload(boundary = boundary)

    def load(self, boundary):
        """反传：取回边界（没有提前上传时现在上传），计算流等上传完成，返回显存张量。"""
        if boundary.device_tensor is None:
            self._upload(boundary = boundary)
        torch.cuda.current_stream(device = self.device).wait_event(event = boundary.uploaded)
        tensor = boundary.device_tensor
        boundary.device_tensor = None
        if self.saved.get(boundary.index) is boundary:
            del self.saved[boundary.index]
        return tensor

    def recycle(self, tensor):
        """这一层的反传内核都已排队，显存缓冲放回缓冲池；之后的上传会先等计算流（见 _upload）。"""
        key = (tuple(tensor.shape), tensor.dtype)
        if key not in self.pool:
            # 批次形状每步都变，只保留当前形状的缓冲。
            self.pool = {key: []}
        if not self.pool[key]:
            self.pool[key].append(tensor)


class _CheckpointBlock(torch.autograd.Function):
    """一层的梯度检查点：前向不保存中间结果，只保存输入；反传时（流式加载这一层后）重算并求输入的梯度。

    激活值卸载提前上传时（stack.boundary_store() 不为 None），输入交给 BoundaryOffload 搬到主机内存，
    不经过 save_for_backward；反传取回这一层的输入后，提前上传前一层的输入。
    """

    @staticmethod
    def forward(ctx, hidden_states, stack, index, forward, args, kwargs):
        store = stack.boundary_store()
        ctx.boundary = None
        if store is not None and hidden_states.is_contiguous():
            ctx.boundary = store.save(index = index, tensor = hidden_states)
        else:
            ctx.save_for_backward(hidden_states)
        ctx.stack, ctx.index, ctx.forward, ctx.args, ctx.kwargs = stack, index, forward, args, kwargs
        stack.enter(index = index, direction = 1)
        output = forward(hidden_states, *args, **kwargs)
        stack.leave(index = index)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        stack, index = ctx.stack, ctx.index
        stack.enter(index = index, direction = -1)
        boundary = ctx.boundary
        if boundary is not None:
            ctx.boundary = None
            hidden_states = boundary.store.load(boundary = boundary)
            boundary.store.prefetch(index = index - 1)
        else:
            (hidden_states,) = ctx.saved_tensors
        with torch.enable_grad():
            inputs = hidden_states.detach().requires_grad_(requires_grad = True)
            output = ctx.forward(inputs, *ctx.args, **ctx.kwargs)
        torch.autograd.backward(tensors = output, grad_tensors = grad_output)
        if boundary is not None:
            boundary.store.recycle(tensor = hidden_states)
        stack.leave(index = index, backward = True)
        return inputs.grad, None, None, None, None, None


class FrozenBlocks:
    """接管一组层（nn.ModuleList）的前向，按设置做梯度检查点、块流式加载和层边界的卸载。

    每层的 forward 换成一个包装：需要梯度时，末尾 checkpoint 个层走 _CheckpointBlock（checkpoint 为 True 时
    是全部层），其余层照常计算并保存中间结果；不需要梯度时（音频塔、dev 评估）直接计算。流式加载的层在计算前
    上传、计算后释放，并提前上传遍历方向上的下一层。需要梯度时流式加载的层必须在做检查点的层之内。
    trainable 为真时（全量微调的语言模型）流式加载的是需要更新的参数，反传算完一层后把它的梯度搬到主机内存。
    boundaries 是 BoundaryOffload，在 offloading 为真的前向中（Applied.forward_context）接管做检查点的层的输入。
    """

    def __init__(
        self,
        blocks,
        checkpoint = False,
        swap_count = 0,
        device = "cuda",
        trainable = False,
        boundaries = None,
    ):
        self.blocks = blocks
        if checkpoint is True:
            checkpoint = len(blocks)
        self.checkpoint = max(0, min(len(blocks), int(checkpoint or 0)))
        # 下标不小于 checkpoint_from 的层做梯度检查点。
        self.checkpoint_from = len(blocks) - self.checkpoint
        self.swap = None
        if swap_count > 0:
            self.swap = BlockSwap(
                blocks = blocks,
                count = swap_count,
                device = device,
                trainable = trainable,
            )
        self.boundaries = boundaries
        self.offloading = False
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

    def leave(self, index, backward = False):
        if self.swap is None:
            return
        if backward and self.swap.trainable and index in self.swap.params:
            self.swap.offload_grads(index = index)
        self.swap.release(index = index)

    def boundary_store(self):
        """当前前向中接管层边界的 BoundaryOffload，不卸载时为 None。"""
        return self.boundaries if self.offloading else None

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

    def streamed_parameters(self):
        """流式加载的需要更新的参数：[(参数, 主机权重, 主机梯度缓冲)]，按层号和参数顺序。"""
        if self.swap is None or not self.swap.trainable:
            return []
        return [
            (parameter, host, grad)
            for index in sorted(self.swap.params)
            for parameter, host, grad in zip(self.swap.params[index], self.swap.host[index], self.swap.host_grads[index])
            if grad is not None
        ]

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
