"""JSONL 读写与分块的小工具，只依赖标准库。"""

import hashlib
import json
import os
from pathlib import Path


def read_jsonl(path: Path) -> list:
    """读取 JSONL，跳过空行。文件不存在时返回空列表。"""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(mode = "r", encoding = "utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl_atomic(path: Path, rows: list) -> None:
    """先写临时文件再改名，保证读到的文件要么完整要么不存在。"""
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open(mode = "w", encoding = "utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii = False, sort_keys = False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(src = tmp, dst = path)


def write_json_atomic(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii = False, indent = 2) + "\n", encoding = "utf-8")
    os.replace(src = tmp, dst = path)


def chunk_name(sample_ids: list) -> str:
    """块文件名由块内全部 sample_id 的哈希决定：同一批样本总是落在同一个文件。"""
    digest = hashlib.sha1("\n".join(sample_ids).encode("utf-8")).hexdigest()
    return f"chunk-{digest[:12]}.jsonl"


def split_chunks(items: list, size: int) -> list:
    size = max(1, int(size))
    return [items[start:start + size] for start in range(0, len(items), size)]


def load_chunk_dir(directory: Path) -> dict:
    """读取目录下所有块文件，返回 sample_id（及可选的角色）到记录的映射。"""
    records = {}
    directory = Path(directory)
    if not directory.exists():
        return records
    for path in sorted(directory.glob("chunk-*.jsonl")):
        for row in read_jsonl(path = path):
            records[row["sample_id"]] = row
    return records
