"""回放文本的存放格式：每组每个划分一个目录，生成结果按 shard 写入，二进制数组旁边配 JSONL 索引。

目录结构（`<回放目录>/<组>/<划分>/`）：

    prompts.jsonl                  # 提示清单，prompts 生成，一行一条，行号即 prompt_index
    raw/shard-<编号>.jsonl         # 生成结果的索引，一行一个样本
    raw/shard-<编号>.ids.i32       # 每个生成 token 的 id（int32）
    raw/shard-<编号>.lp.f32        # 每个生成 token 自身的对数概率（float32）
    raw/shard-<编号>.topk_ids.i32  # 每个生成 token 位置上参考模型概率最高的 20 个 token id（int32，每行 20 个）
    raw/shard-<编号>.topk_lp.f16   # 对应的 20 个对数概率（float16）
    samples.jsonl                  # finalize 写出的最终索引：截断位置、文本、语言、退化统计

四个二进制文件按生成 token 平铺，第 i 个生成 token 的数据在各文件的第 i 个位置（top-20 两个文件的第 i 行）。
索引中每个样本由若干片段（`pieces`）依次拼成：`{"ids": [...]}` 是直接给定的 token（提示、聊天模板），
`{"gen": 角色, "offset": o, "length": n, ...}` 是模型生成的 token，位于本 shard 二进制数组的 [o, o + n)。
对数概率是参考模型在温度 1 下的原始对数概率（log softmax），与采样分布一致。

每次运行写一个新的 shard，样本完成时先写二进制、后写索引行。启动时 `repair` 按索引截掉多出的二进制尾部，
并丢弃写到一半的最后一行，因此进程中途退出后可以直接续跑。
"""

import json
from pathlib import Path

import numpy as np

TOP_K = 20
SUFFIXES = {
    "ids": (".ids.i32", np.int32, 1),
    "lp": (".lp.f32", np.float32, 1),
    "topk_ids": (".topk_ids.i32", np.int32, TOP_K),
    "topk_lp": (".topk_lp.f16", np.float16, TOP_K),
}


def shard_paths(raw_dir, number):
    base = Path(raw_dir) / f"shard-{number:05d}"
    return base.with_suffix(".jsonl"), {name: Path(str(base) + suffix) for name, (suffix, _, _) in SUFFIXES.items()}


def list_shards(raw_dir):
    return sorted(int(path.stem.split("-")[1]) for path in Path(raw_dir).glob("shard-*.jsonl"))


def gen_tokens(record):
    """样本中生成片段占用的二进制长度之和。"""
    return sum(piece["length"] for piece in record["pieces"] if "gen" in piece)


def read_index(path):
    """读取 shard 索引；写到一半的最后一行（没有换行或不是完整 JSON）忽略。"""
    records = []
    with open(path, encoding = "utf-8") as handle:
        for line in handle:
            if not line.endswith("\n"):
                break
            try:
                records.append(json.loads(line))
            except ValueError:
                break
    return records


def repair(raw_dir):
    """让每个 shard 的二进制长度与索引一致：截掉索引之外的尾部，重写不完整的索引。"""
    for number in list_shards(raw_dir = raw_dir):
        index_path, paths = shard_paths(raw_dir = raw_dir, number = number)
        records = read_index(path = index_path)
        with open(index_path, encoding = "utf-8") as handle:
            n_lines = handle.read().count("\n")
        if n_lines != len(records):
            with open(index_path, "w", encoding = "utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii = False) + "\n")
        n_tokens = max((piece["offset"] + piece["length"] for record in records for piece in record["pieces"] if "gen" in piece), default = 0)
        for name, path in paths.items():
            _, dtype, width = SUFFIXES[name]
            expected = n_tokens * width * np.dtype(dtype).itemsize
            if path.exists() and path.stat().st_size > expected:
                with open(path, "r+b") as handle:
                    handle.truncate(expected)


def read_all_records(raw_dir):
    """全部 shard 的索引，每条附上 shard 编号。"""
    records = []
    for number in list_shards(raw_dir = raw_dir):
        index_path, _ = shard_paths(raw_dir = raw_dir, number = number)
        for record in read_index(path = index_path):
            record["shard"] = number
            records.append(record)
    return records


class ShardWriter:
    """追加写一个新 shard。"""

    def __init__(self, raw_dir):
        raw_dir = Path(raw_dir)
        raw_dir.mkdir(parents = True, exist_ok = True)
        existing = list_shards(raw_dir = raw_dir)
        self.number = (existing[-1] + 1) if existing else 0
        self.index_path, paths = shard_paths(raw_dir = raw_dir, number = self.number)
        self.handles = {name: open(path, "ab") for name, path in paths.items()}
        self.index = open(self.index_path, "a", encoding = "utf-8")
        self.n_tokens = 0

    def add_gen(self, ids, lp, topk_ids, topk_lp):
        """写入一段生成 token，返回它在本 shard 中的起点。"""
        offset = self.n_tokens
        np.asarray(ids, dtype = np.int32).tofile(self.handles["ids"])
        np.asarray(lp, dtype = np.float32).tofile(self.handles["lp"])
        np.asarray(topk_ids, dtype = np.int32).reshape(-1, TOP_K).tofile(self.handles["topk_ids"])
        np.asarray(topk_lp, dtype = np.float16).reshape(-1, TOP_K).tofile(self.handles["topk_lp"])
        self.n_tokens += len(ids)
        return offset

    def add_record(self, record):
        for handle in self.handles.values():
            handle.flush()
        self.index.write(json.dumps(record, ensure_ascii = False) + "\n")
        self.index.flush()

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.index.close()


class ShardReader:
    """按需读取 shard 的二进制数组（内存映射）。"""

    def __init__(self, raw_dir):
        self.raw_dir = Path(raw_dir)
        self.cache = {}

    def arrays(self, number):
        if number not in self.cache:
            _, paths = shard_paths(raw_dir = self.raw_dir, number = number)
            arrays = {}
            for name, path in paths.items():
                _, dtype, width = SUFFIXES[name]
                data = np.memmap(path, dtype = dtype, mode = "r") if path.stat().st_size else np.zeros(0, dtype = dtype)
                arrays[name] = data.reshape(-1, width) if width > 1 else data
            self.cache[number] = arrays
        return self.cache[number]

    def piece(self, number, piece, name = "ids"):
        data = self.arrays(number = number)[name]
        return data[piece["offset"]: piece["offset"] + piece["length"]]
