"""音频分片：把训练要读的音频文件按训练读取的顺序打包成约 1 GB 的 tar，训练时按位置读出原始字节再解码（任务 C1）。

在 Google Colab 上训练时，v2b 的约 480 万个 FLAC 文件（614 GB）放不进本地磁盘，Google Drive 上大量小文件的上传、
列目录、打开和随机读取又都很慢。分片把它们打包成几百个 tar 文件放在 Drive 上；训练时只在本地保留一段滚动窗口，
用完就删，后面的整段顺序复制下来（pipeline/colab/shard_window.py），不从 Drive 随机读取。

分片是标准的 POSIX tar（不压缩），成员是原来的 FLAC 文件，名字是相对 member_root 的路径，字节原样写入，不重新
编码、不截断。每个成员前有 512 字节的头（名字很长或含非 ASCII 字符时多一段 PAX 扩展头），数据补齐到 512 字节，
平均每个文件多约 1 KB（约 0.8%）。索引记录每个文件的数据在分片中的起点和长度，读取时一次 os.pread 取出原始
字节；索引丢失时也可以用 tar 列出或解开分片。

分片有两种排列方式（索引中的 order）。按路径排列（path）时全部文件按 audio_path 排序后依次装入分片，与训练计划
无关，适合挂载成目录、随机读取的存储（例如用 gcsfuse 挂载的 Google Cloud Storage 存储桶）。按训练计划排列（plan）
时分片的顺序跟随训练读取的顺序，适合在本地维护滚动窗口。训练计划（sampling.EpochPlan）只由训练配置决定，可以
提前算出每一步读哪些文件：

- 常驻分片（pool）：这一轮中读取不止一次的文件，例如被加权多抽的无人声片段、虚拟切分后在不同步数读取的几段
  所共用的整段录音。v2b 约 48 GB，每次在 Colab 上开始训练前整体复制到本地，训练期间一直保留。
- 顺序分片（stream）：只读一次的文件，按读取它的步数排列，每个分片记录它服务的步数范围（first_step、last_step），
  窗口管理（pipeline/colab/shard_window.py）据此决定先复制哪些、哪些可以删除。

步数用从 0 开始的下标：训练完成 s 步（progress.json 和检查点中的步数）后，下一个读取的是下标为 s 的步；
一步含 grad_accum 个微批。

索引由同一目录下的两个文件组成：

- <名称>.json：格式名、排列方式 order、member_root、成员的修改时间、目标分片大小、来源清单的配置 source、
  训练计划（plan，按路径排列时为 null：轮次、决定计划的配置项、按读取顺序算出的摘要 digest、这一轮的步数范围、
  索引覆盖的步数范围）和分片列表（文件名、种类 pool / stream / path、文件数、字节数，顺序分片另有步数范围）。
- <名称>.tsv.gz：每个音频文件一行，依次是 audio_path、分片编号、数据在分片中的起点（字节）和长度（字节）。

audio_path 与 data.read_manifest 解析后的路径完全相同，训练时直接用清单行的 audio_path 查索引。只为一段步数
做速度测试时，索引可以只覆盖这段步数（plan.steps）：顺序分片与完整索引相同，常驻分片只含这段步数用到的文件。

训练时（train.py 的 --audio-shards 等选项）attach_shards 在主进程中读入索引，给每行清单加上 audio_shard =
(分片编号, 数据起点, 长度)；train.py 核对索引的训练计划与本次训练一致，不一致就不开始训练。数据读取进程只收到
分片名单和查找目录，用 ShardReader 按需打开分片、读出字节，交给 data.load_audio 解码。
"""

import gzip
import hashlib
import json
import os
import sys
import tarfile
import time
from collections import OrderedDict
from pathlib import Path

FORMAT_NAME = "qwen3.5-asr-audio-shards/2"
TAR_FORMAT = tarfile.PAX_FORMAT
TAR_ENCODING = "utf-8"
BLOCK = tarfile.BLOCKSIZE
RECORD = tarfile.RECORDSIZE
# 每个进程最多同时打开的顺序分片数。顺序分片按步数排列，一个读取进程只会用到当前附近的一两个；窗口管理删掉的
# 分片要等打开它的进程关闭后才真正释放磁盘空间，所以旧的顺序分片及时关闭。常驻分片一直打开。
STREAM_OPEN = 3
# 按路径排列的分片被随机读取，每个进程最多同时打开这么多个（v2b 约 620 个，通常全部保持打开）。
PATH_OPEN = 1024
# 读分片出错（例如 Google Drive 挂载点暂时出错）时的尝试次数，两次之间等待 0.5 秒、1 秒……
READ_ATTEMPTS = 3


def member_info(name, size, mtime):
    """tar 成员的头信息：普通文件，权限 644，属主为空，修改时间统一为 mtime。"""
    info = tarfile.TarInfo(name = name)
    info.size = size
    info.mtime = int(mtime)
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def member_header(name, size, mtime):
    """这个成员写入 tar 时的头（含可能的 PAX 扩展头）。"""
    return member_info(name = name, size = size, mtime = mtime).tobuf(
        format = TAR_FORMAT,
        encoding = TAR_ENCODING,
        errors = "surrogateescape",
    )


def padded(size):
    """成员数据补齐到 512 字节后的长度。"""
    return -(-size // BLOCK) * BLOCK


def archive_bytes(position):
    """最后一个成员结束于 position 时整个 tar 的大小：结尾两个全零块，再补齐到 tarfile 的记录大小。"""
    return -(-(position + 2 * BLOCK) // RECORD) * RECORD


def layout(members, shard_bytes, mtime):
    """按顺序把成员装进 tar 分片，返回 (locations, shards)。

    members 是 [(成员名, 字节数), ...]。locations[i] 是第 i 个成员的 (第几个分片, 数据起点)，shards[k] 是第 k 个
    分片的 (成员数, tar 大小)。再装一个成员会让 tar 超过 shard_bytes 时换下一个分片；一个分片至少装一个成员。
    """
    locations = []
    shards = []
    number = 0
    position = 0
    count = 0
    for name, size in members:
        header = len(member_header(name = name, size = size, mtime = mtime))
        if count and archive_bytes(position = position + header + padded(size = size)) > shard_bytes:
            shards.append((count, archive_bytes(position = position)))
            number += 1
            position = 0
            count = 0
        locations.append((number, position + header))
        position += header + padded(size = size)
        count += 1
    if count:
        shards.append((count, archive_bytes(position = position)))
    return locations, shards


def member_name(path, root):
    """音频文件在 tar 中的成员名：相对 member_root 的路径。"""
    return Path(os.path.relpath(path = path, start = root)).as_posix()


def write_index(
    directory,
    name,
    header,
    entries,
):
    """写出 <name>.tsv.gz 和 <name>.json（先写条目，再写 json），返回 json 的路径。

    entries 依次给出 (audio_path, 分片编号, 数据起点, 长度)。
    """
    directory = Path(directory)
    directory.mkdir(parents = True, exist_ok = True)
    header = dict(header, format = FORMAT_NAME, entries = f"{name}.tsv.gz")
    target = directory / header["entries"]
    tmp = target.with_name(name = target.name + ".tmp")
    with gzip.open(
        filename = tmp,
        mode = "wt",
        encoding = "utf-8",
        compresslevel = 6,
    ) as handle:
        for path, number, offset, size in entries:
            if "\t" in path or "\n" in path:
                raise ValueError(f"音频路径中有制表符或换行，无法写入索引：{path!r}")
            handle.write(f"{path}\t{number}\t{offset}\t{size}\n")
    tmp.replace(target = target)
    index = directory / f"{name}.json"
    tmp = index.with_name(name = index.name + ".tmp")
    tmp.write_text(data = json.dumps(obj = header, ensure_ascii = False, indent = 1) + "\n", encoding = "utf-8")
    tmp.replace(target = index)
    return index


def read_header(index_path):
    """读取索引的 json 部分，附上它自己的路径 index_path。"""
    index_path = Path(index_path)
    header = json.loads(s = index_path.read_text(encoding = "utf-8"))
    if header.get("format") != FORMAT_NAME:
        raise ValueError(f"{index_path} 不是这个版本的音频分片索引（format = {header.get('format')!r}）")
    header["index_path"] = str(index_path)
    return header


def iter_entries(header):
    """依次给出索引中的 (audio_path, 分片编号, 数据起点, 长度)，同一分片的条目按起点排列。"""
    entries = Path(header["index_path"]).with_name(name = header["entries"])
    with gzip.open(filename = entries, mode = "rt", encoding = "utf-8") as handle:
        for line in handle:
            path, number, offset, size = line.rstrip("\n").split(sep = "\t")
            yield path, int(number), int(offset), int(size)


def load_locations(header):
    """把索引读成 {audio_path: (分片编号, 数据起点, 长度)}。"""
    return {path: (number, offset, size) for path, number, offset, size in iter_entries(header = header)}


def parse_dirs(spec):
    """把「目录[:目录...]」拆成目录列表，空项忽略。"""
    if not spec:
        return []
    return [Path(part) for part in str(spec).split(sep = os.pathsep) if part]


def complete_path(directory, shard):
    """directory 中这个分片（索引中的一项）的路径；文件不存在或大小不对（还没写完、没复制完）时返回 None。"""
    path = Path(directory) / shard["name"]
    try:
        size = path.stat().st_size
    except OSError:
        return None
    return path if size == shard["bytes"] else None


def local_shards(header, dirs):
    """在 dirs 中有完整文件的分片编号集合。"""
    return {
        number for number, shard in enumerate(header["shards"])
        if any(complete_path(directory = directory, shard = shard) is not None for directory in dirs)
    }


def format_ranges(numbers):
    """[0, 1, 2, 5, 7, 8] → "0-2,5,7-8"。"""
    parts = []
    start = previous = None
    for number in sorted(set(numbers)):
        if start is None:
            start = previous = number
        elif number == previous + 1:
            previous = number
        else:
            parts.append(str(start) if start == previous else f"{start}-{previous}")
            start = previous = number
    if start is not None:
        parts.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(parts)


def plan_batches(plan, epoch, grad_accum):
    """按训练读取的顺序给出 (步的下标, 这个微批的清单行号列表)。

    plan 是 sampling.EpochPlan。轮次 epoch 之前各轮的微批数从同一个计划算出，步的下标在各轮之间连续。
    """
    first = sum(len(plan.position_batches(epoch = earlier)[1]) for earlier in range(epoch))
    occurrences, batches = plan.position_batches(epoch = epoch)
    for number, batch in enumerate(batches):
        yield (first + number) // grad_accum, [int(occurrences[position]) for position in batch]


def digest_lines(step, paths):
    """计入摘要的内容：一个微批中每次读取一行「步的下标、音频路径」。"""
    return "".join(f"{step}\t{path}\n" for path in paths).encode(encoding = "utf-8")


def plan_digest(
    plan,
    rows,
    epoch,
    grad_accum,
):
    """这一轮按读取顺序的「步的下标、音频路径」序列的 SHA-256，用来核对训练与分片按同一个计划排列。"""
    digest = hashlib.sha256()
    for step, indices in plan_batches(plan = plan, epoch = epoch, grad_accum = grad_accum):
        digest.update(digest_lines(step = step, paths = [rows[index]["audio_path"] for index in indices]))
    return digest.hexdigest()


class ShardReader:
    """按 (分片编号, 数据起点, 长度) 读出音频文件的原始字节。

    分片在第一次用到时按 dirs 的顺序查找大小正确的文件（本地的窗口目录、常驻目录），都没有时到 fallback（例如
    Google Drive 上的分片目录）中找，并在标准错误上记一行 [shard-fallback]，每个进程每个分片只记一次。每个进程
    （主进程或 DataLoader 的读取进程）各自打开分片：常驻分片打开后一直保留，顺序分片最多同时打开 STREAM_OPEN 个，
    超出时关闭最久没用的一个。读取用 os.pread，不移动文件位置。读取出错时关掉这个分片重新查找、打开再读，共尝试
    READ_ATTEMPTS 次，仍然失败就抛出异常，由调用方当作读不出的音频处理（[bad-audio]）。
    """

    def __init__(self, shards, dirs, fallback = None):
        """shards 是索引中的分片列表（每项含 name、bytes、kind）。"""
        self.names = [shard["name"] for shard in shards]
        self.sizes = [int(shard["bytes"]) for shard in shards]
        self.kinds = [shard["kind"] for shard in shards]
        self.dirs = [str(directory) for directory in dirs]
        self.fallback = str(fallback) if fallback else None
        self._reset()

    def _reset(self):
        self._pool = {}
        self._stream = OrderedDict()
        self._warned = set()
        self._pid = os.getpid()
        self._limit_raised = False

    def __getstate__(self):
        return {
            "names": self.names,
            "sizes": self.sizes,
            "kinds": self.kinds,
            "dirs": self.dirs,
            "fallback": self.fallback,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._reset()

    def _try_open(self, directory, number):
        path = os.path.join(directory, self.names[number])
        try:
            descriptor = os.open(path = path, flags = os.O_RDONLY)
        except FileNotFoundError:
            return None
        if os.fstat(fd = descriptor).st_size != self.sizes[number]:
            os.close(fd = descriptor)
            return None
        return descriptor

    def _open(self, number):
        for directory in self.dirs:
            descriptor = self._try_open(directory = directory, number = number)
            if descriptor is not None:
                return descriptor
        if self.fallback is not None:
            descriptor = self._try_open(directory = self.fallback, number = number)
            if descriptor is not None:
                if number not in self._warned:
                    self._warned.add(number)
                    print(f"[shard-fallback] {self.names[number]} 不在本地，从 {self.fallback} 读取", file = sys.stderr, flush = True)
                return descriptor
        places = self.dirs + ([self.fallback] if self.fallback else [])
        raise FileNotFoundError(f"分片 {self.names[number]} 不在 {places} 中")

    def _raise_open_limit(self):
        """把本进程可打开文件数的软上限提到硬上限（最多 65536），按路径排列的分片要同时打开几百个。"""
        self._limit_raised = True
        try:
            import resource

            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            wanted = 65536 if hard == resource.RLIM_INFINITY else min(hard, 65536)
            if soft != resource.RLIM_INFINITY and soft < wanted:
                resource.setrlimit(resource.RLIMIT_NOFILE, (wanted, hard))
        except (ImportError, ValueError, OSError):
            pass

    def _descriptor(self, number):
        if self._pid != os.getpid():
            # fork 出的子进程继承了父进程打开的描述符：关掉，各进程使用自己打开的描述符。
            self.close()
            self._reset()
        if not self._limit_raised:
            self._raise_open_limit()
        kind = self.kinds[number]
        table = self._pool if kind == "pool" else self._stream
        descriptor = table.get(number)
        if descriptor is not None:
            if kind != "pool":
                self._stream.move_to_end(key = number)
            return descriptor
        descriptor = self._open(number = number)
        table[number] = descriptor
        # 顺序分片和按路径排列的分片放在同一张表里按最近使用关闭；两种不会出现在同一个索引中。
        limit = STREAM_OPEN if kind == "stream" else PATH_OPEN
        while len(self._stream) > limit:
            _, oldest = self._stream.popitem(last = False)
            os.close(fd = oldest)
        return descriptor

    def _forget(self, number):
        for table in (self._pool, self._stream):
            descriptor = table.pop(number, None)
            if descriptor is not None:
                try:
                    os.close(fd = descriptor)
                except OSError:
                    pass

    def _read_once(self, number, offset, size):
        descriptor = self._descriptor(number = number)
        chunks = []
        done = 0
        while done < size:
            chunk = os.pread(descriptor, size - done, offset + done)
            if not chunk:
                raise OSError(f"{self.names[number]} 在第 {offset} 字节处只读到 {done}/{size} 字节")
            chunks.append(chunk)
            done += len(chunk)
        return chunks[0] if len(chunks) == 1 else b"".join(chunks)

    def read(self, location):
        """location 是 (分片编号, 数据起点, 长度)，返回这段字节；为 None 时说明索引中没有这个音频。"""
        if location is None:
            raise FileNotFoundError("分片索引中没有这个音频（索引只覆盖部分步数，或按另一份清单生成）")
        number, offset, size = (int(value) for value in location)
        for attempt in range(1, READ_ATTEMPTS + 1):
            try:
                return self._read_once(number = number, offset = offset, size = size)
            except FileNotFoundError:
                raise
            except OSError:
                self._forget(number = number)
                if attempt == READ_ATTEMPTS:
                    raise
                time.sleep(0.5 * attempt)

    def close(self):
        for table in (self._pool, self._stream):
            while table:
                _, descriptor = table.popitem()
                try:
                    os.close(fd = descriptor)
                except OSError:
                    pass


def attach_shards(
    rows,
    header,
    dirs,
    fallback = None,
):
    """给索引中有的清单行加上 audio_shard = (分片编号, 数据起点, 长度)，返回 (ShardReader, 概况)。

    这一轮不读的行（抽样权重为 0 或没有抽中），以及只覆盖部分步数的索引中这段步数以外的行，索引中没有，不加。
    概况给出加上位置的行数、常驻分片和顺序分片在本地（dirs）有几个完整文件，以及备用目录。
    """
    locations = load_locations(header = header)
    located = 0
    for row in rows:
        location = locations.get(row["audio_path"])
        if location is not None:
            row["audio_shard"] = location
            located += 1
    del locations
    local = local_shards(header = header, dirs = dirs)
    kinds = [shard["kind"] for shard in header["shards"]]
    summary = {
        "index": header["index_path"],
        "order": header["order"],
        "steps": header["plan"]["steps"] if header["plan"] else None,
        "rows": len(rows),
        "rows_in_index": located,
        "dirs": [str(directory) for directory in dirs],
        "fallback": str(fallback) if fallback else None,
    }
    for kind in sorted(set(kinds)):
        summary[f"{kind}_in_dirs"] = f"{sum(1 for number in local if kinds[number] == kind)}/{kinds.count(kind)}"
    return ShardReader(shards = header["shards"], dirs = dirs, fallback = fallback), summary
