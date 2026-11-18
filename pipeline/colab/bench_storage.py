"""比较几处分片存储的读取速度：Google Drive、gcsfuse 挂载的存储桶、本地磁盘（任务 C1）。

每个 --dir 是「名称=目录」，目录中放着索引里的分片（文件名与索引一致，大小正确的才算）。对每个目录做两项测试：

- copy：把其中 --copy-shards 个分片整个顺序复制到 --scratch-dir，测 MB/s，复制完删掉。shard_window.py 从 Drive
  复制分片、gcloud storage cp 批量复制都是这种读法。--gs「名称=gs://桶/前缀」另用 gcloud storage cp 复制同样的
  分片做对照。
- random：从这个目录已有的分片中随机取 --reads 个音频文件，用 --threads 个线程同时按位置读取（os.pread，与训练时的
  读取方式相同），测每秒请求数、MB/s 和每次读取的延迟分位数（p50、p95、p99）。先打开全部要用的分片，打开耗时
  单独给出。加 --decode 时每个线程读完接着用 torchcodec 解码，另给出平均解码耗时。

开始时从 GCE 元数据服务器读出 Colab 虚拟机所在的可用区（zone），用来判断它与各存储桶是否在同一区域。每项结果
打印一行 JSON，最后打印一张表。本地目录刚复制过的分片可能还在页缓存中，读取会偏快；Drive 的挂载也有自己的缓存，
重复测试时换一个 --seed。

    python -m pipeline.colab.bench_storage --index <索引 json> \\
        --dir drive=<Drive 分片目录> --dir tokyo=/content/gcs/<桶名>/shards --dir local=<本地分片目录> \\
        --gs tokyo=gs://<桶名>/shards --threads 16 --reads 2000
"""

import argparse
import json
import os
import random
import shutil
import statistics
import subprocess
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pipeline.train.shards import complete_path, iter_entries, read_header

GB = 1e9
COPY_CHUNK = 16 * 2 ** 20
METADATA_ZONE = "http://metadata.google.internal/computeMetadata/v1/instance/zone"


def vm_zone():
    """虚拟机所在的可用区（例如 asia-northeast1-b）；不在 Google Cloud 上或读不到时返回 None。"""
    request = urllib.request.Request(url = METADATA_ZONE, headers = {"Metadata-Flavor": "Google"})
    try:
        with urllib.request.urlopen(url = request, timeout = 2) as response:
            # 返回值形如 projects/<项目编号>/zones/<可用区>，只取可用区
            return response.read().decode(encoding = "utf-8").rsplit(sep = "/", maxsplit = 1)[-1]
    except (OSError, ValueError):
        return None


def parse_named(specs):
    """把 ["drive=/a", "tokyo=/b"] 解析成 [("drive", "/a"), ("tokyo", "/b")]。"""
    pairs = []
    for spec in specs:
        name, equals, value = spec.partition("=")
        if not equals or not name or not value:
            raise ValueError(f"应写成「名称=位置」：{spec!r}")
        pairs.append((name, value))
    return pairs


def present_shards(header, directory):
    """directory 中大小正确的分片编号。先列目录，只对列出的文件名核对大小。"""
    try:
        listed = set(os.listdir(path = directory))
    except OSError:
        return []
    return [
        number for number, shard in enumerate(header["shards"])
        if shard["name"] in listed and complete_path(directory = directory, shard = shard) is not None
    ]


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))]


def copy_test(
    header,
    name,
    directory,
    numbers,
    scratch,
):
    """整个顺序复制这些分片，返回结果字典。"""
    scratch.mkdir(parents = True, exist_ok = True)
    total = 0
    per_shard = []
    start = time.perf_counter()
    for number in numbers:
        shard = header["shards"][number]
        target = scratch / shard["name"]
        begin = time.perf_counter()
        with open(file = Path(directory) / shard["name"], mode = "rb") as reader, open(file = target, mode = "wb") as writer:
            shutil.copyfileobj(fsrc = reader, fdst = writer, length = COPY_CHUNK)
        seconds = time.perf_counter() - begin
        target.unlink()
        total += shard["bytes"]
        per_shard.append(round(number = shard["bytes"] / 1e6 / max(seconds, 1e-9), ndigits = 1))
    seconds = time.perf_counter() - start
    return {
        "dir": name,
        "test": "copy",
        "shards": [header["shards"][number]["name"] for number in numbers],
        "gb": round(number = total / GB, ndigits = 2),
        "seconds": round(number = seconds, ndigits = 1),
        "mb_per_s": round(number = total / 1e6 / max(seconds, 1e-9), ndigits = 1),
        "per_shard_mb_per_s": per_shard,
    }


def gcloud_copy_test(
    header,
    name,
    uri,
    numbers,
    scratch,
):
    """用 gcloud storage cp 把同样的分片从存储桶复制到本地，返回结果字典。"""
    scratch.mkdir(parents = True, exist_ok = True)
    total = 0
    start = time.perf_counter()
    for number in numbers:
        shard = header["shards"][number]
        target = scratch / shard["name"]
        completed = subprocess.run(
            args = ["gcloud", "storage", "cp", f"{uri.rstrip('/')}/{shard['name']}", str(target)],
            capture_output = True,
            text = True,
        )
        if completed.returncode != 0:
            return {"dir": name, "test": "gcloud-copy", "error": completed.stderr.strip()[-500:]}
        target.unlink()
        total += shard["bytes"]
    seconds = time.perf_counter() - start
    return {
        "dir": name,
        "test": "gcloud-copy",
        "gb": round(number = total / GB, ndigits = 2),
        "seconds": round(number = seconds, ndigits = 1),
        "mb_per_s": round(number = total / 1e6 / max(seconds, 1e-9), ndigits = 1),
    }


def random_test(
    header,
    name,
    directory,
    entries,
    threads,
    decode,
):
    """多线程随机读取这些条目，返回结果字典。entries 是 [(audio_path, 分片编号, 数据起点, 长度)]。"""
    descriptors = {}
    begin = time.perf_counter()
    for number in sorted({entry[1] for entry in entries}):
        descriptors[number] = os.open(path = os.path.join(directory, header["shards"][number]["name"]), flags = os.O_RDONLY)
    open_seconds = time.perf_counter() - begin
    lock = threading.Lock()
    decode_seconds = []

    def read_one(entry):
        path, number, offset, size = entry
        started = time.perf_counter()
        data = os.pread(descriptors[number], size, offset)
        latency = time.perf_counter() - started
        if len(data) != size:
            raise OSError(f"{path} 只读到 {len(data)}/{size} 字节")
        if decode:
            from pipeline.train.data import load_audio

            started = time.perf_counter()
            load_audio(path = path, data = data)
            with lock:
                decode_seconds.append(time.perf_counter() - started)
        return latency, size

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers = threads) as pool:
        results = list(pool.map(read_one, entries))
    seconds = time.perf_counter() - start
    for descriptor in descriptors.values():
        os.close(fd = descriptor)
    latencies = [latency for latency, _ in results]
    total = sum(size for _, size in results)
    report = {
        "dir": name,
        "test": "random",
        "reads": len(results),
        "threads": threads,
        "shards_opened": len(descriptors),
        "open_ms_mean": round(number = 1e3 * open_seconds / max(len(descriptors), 1), ndigits = 1),
        "seconds": round(number = seconds, ndigits = 2),
        "requests_per_s": round(number = len(results) / max(seconds, 1e-9), ndigits = 1),
        "mb_per_s": round(number = total / 1e6 / max(seconds, 1e-9), ndigits = 1),
        "mean_kb": round(number = total / 1e3 / max(len(results), 1), ndigits = 1),
        "p50_ms": round(number = 1e3 * statistics.median(data = latencies), ndigits = 1),
        "p95_ms": round(number = 1e3 * percentile(values = latencies, fraction = 0.95), ndigits = 1),
        "p99_ms": round(number = 1e3 * percentile(values = latencies, fraction = 0.99), ndigits = 1),
    }
    if decode_seconds:
        report["decode_ms_mean"] = round(number = 1e3 * statistics.fmean(data = decode_seconds), ndigits = 2)
    return report


def table(results):
    """把结果排成一张便于阅读的表。"""
    lines = [f"{'目录':<12}{'测试':<13}{'MB/s':>9}{'请求/s':>10}{'p50 ms':>9}{'p95 ms':>9}{'p99 ms':>9}"]
    for result in results:
        if result.get("test") == "zone":
            continue
        if "error" in result:
            lines.append(f"{result['dir']:<12}{result['test']:<13}  出错：{result['error'][:60]}")
            continue
        lines.append(
            f"{result['dir']:<12}{result['test']:<13}{result.get('mb_per_s', ''):>9}{result.get('requests_per_s', ''):>10}"
            f"{result.get('p50_ms', ''):>9}{result.get('p95_ms', ''):>9}{result.get('p99_ms', ''):>9}"
        )
    return "\n".join(lines)


def run(args):
    header = read_header(index_path = args.index)
    results = []

    def emit(result):
        results.append(result)
        print(json.dumps(obj = result, ensure_ascii = False), flush = True)

    emit({"test": "zone", "zone": vm_zone()})
    dirs = parse_named(specs = args.dir)
    rng = random.Random(x = args.seed)
    copied = {}
    present = {name: present_shards(header = header, directory = directory) for name, directory in dirs}
    wanted = {number for numbers in present.values() for number in numbers}
    # 只读一遍索引，留下各目录已有分片中的条目
    entries_by_shard = {}
    for path, number, offset, size in iter_entries(header = header):
        if number in wanted:
            entries_by_shard.setdefault(number, []).append((path, number, offset, size))
    for name, directory in dirs:
        numbers = present[name]
        if not numbers:
            emit({"dir": name, "test": "copy", "error": f"{directory} 中没有索引里的完整分片"})
            continue
        if args.copy_shards > 0:
            chosen = sorted(rng.sample(population = numbers, k = min(args.copy_shards, len(numbers))))
            copied[name] = chosen
            emit(copy_test(
                header = header,
                name = name,
                directory = directory,
                numbers = chosen,
                scratch = Path(args.scratch_dir),
            ))
        if args.reads > 0:
            population = [entry for number in numbers for entry in entries_by_shard[number]]
            entries = rng.sample(population = population, k = min(args.reads, len(population)))
            emit(random_test(
                header = header,
                name = name,
                directory = directory,
                entries = entries,
                threads = args.threads,
                decode = args.decode,
            ))
    for name, uri in parse_named(specs = args.gs):
        numbers = copied.get(name) or list(range(min(args.copy_shards, len(header["shards"]))))
        emit(gcloud_copy_test(
            header = header,
            name = name,
            uri = uri,
            numbers = numbers,
            scratch = Path(args.scratch_dir),
        ))
    print(table(results = results), flush = True)
    return results


def parse_args(argv = None):
    parser = argparse.ArgumentParser(description = "比较几处分片存储的顺序复制和随机读取速度")
    parser.add_argument("--index", required = True, help = "分片索引 json")
    parser.add_argument(
        "--dir",
        action = "append",
        default = [],
        help = "「名称=目录」，可重复",
    )
    parser.add_argument(
        "--gs",
        action = "append",
        default = [],
        help = "「名称=gs://桶/前缀」，用 gcloud storage cp 测复制，可重复",
    )
    parser.add_argument("--scratch-dir", default = None, help = "复制测试写到这里（本地磁盘），测完删除；默认是系统临时目录")
    parser.add_argument(
        "--copy-shards",
        type = int,
        default = 3,
        help = "每处复制几个分片；0 不测",
    )
    parser.add_argument(
        "--reads",
        type = int,
        default = 2000,
        help = "随机读取的条目数；0 不测",
    )
    parser.add_argument(
        "--threads",
        type = int,
        default = 16,
        help = "随机读取的并发线程数",
    )
    parser.add_argument("--decode", action = "store_true", help = "随机读取后同时解码")
    parser.add_argument(
        "--seed",
        type = int,
        default = 0,
        help = "换一个种子可以避开上次读过、已在缓存中的部分",
    )
    return parser.parse_args(args = argv)


def main(argv = None):
    args = parse_args(argv = argv)
    if args.scratch_dir is None:
        with tempfile.TemporaryDirectory(prefix = "bench-storage-") as scratch:
            args.scratch_dir = scratch
            return run(args = args)
    return run(args = args)


if __name__ == "__main__":
    main()
