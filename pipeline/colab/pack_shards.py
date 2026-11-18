"""把训练音频打包成 tar 分片，供 Colab 上训练读取（任务 C1）。

分片格式和两种排列方式见 pipeline/train/shards.py。音频文件的字节原样写入 tar，不重新编码、不截断、不改变长度。
--order plan（默认）按训练计划排列，供 Google Drive 加本地滚动窗口使用（shard_window.py）；--order path 按路径
排列，供随机读取的存储使用（例如 gcsfuse 挂载的存储桶）。训练计划由训练配置决定，这里直接调用训练程序自己的函数
（train.training_rows、data.training_targets、train.training_weights、train.make_batch_stream）算出每一步读哪些文件，
不另写一套规则。

训练配置有两种给法：--run-config 指向已有训练的 train_config.json（续训时最可靠），或者在 -- 之后写出与
run.sh 相同的 train.py 参数。决定计划的配置项见 train.PLAN_KEYS，其余参数不影响分片。

    # 1. 生成索引（读清单、算训练计划、取文件大小，不写分片）
    python -m pipeline.colab.pack_shards index --out-dir <分片目录> --run-config <输出目录>/train_config.json
    python -m pipeline.colab.pack_shards index --out-dir <分片目录> -- \\
        --train-manifest <训练清单> --target original --batch-size 16 --max-duration-s 30 --max-batch-seconds 360
    # 只覆盖一段步数的索引（速度测试用）：顺序分片与完整索引相同，常驻分片只含这段步数用到的文件
    python -m pipeline.colab.pack_shards index --out-dir <分片目录> --name index-30000-40000 --steps 30000-40000 --run-config ...
    # 按路径排列
    python -m pipeline.colab.pack_shards index --out-dir <分片目录> --order path --run-config ...
    # 2. 分批生成分片；已经完整的分片（大小与索引一致）跳过
    python -m pipeline.colab.pack_shards pack --index <分片目录>/index.json --pool
    python -m pipeline.colab.pack_shards pack --index <分片目录>/index.json --shards 48-147
    python -m pipeline.colab.pack_shards pack --index <分片目录>/index.json --steps 30000-40000
    # 3. 核对：tar 成员表与索引一致，随机抽查的成员字节与原文件一致
    python -m pipeline.colab.pack_shards verify --index <分片目录>/index.json --samples 500
    # 查看各目录中哪些分片已完整
    python -m pipeline.colab.pack_shards status --index <分片目录>/index.json --dirs <目录>[:<目录>...]
"""

import argparse
import hashlib
import io
import json
import os
import random
import shutil
import sys
import tarfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pipeline.train.shards import (
    TAR_ENCODING,
    TAR_FORMAT,
    complete_path,
    digest_lines,
    format_ranges,
    iter_entries,
    layout,
    member_header,
    member_info,
    member_name,
    parse_dirs,
    plan_batches,
    read_header,
    write_index,
)

GB = 1e9
# 一次提交给线程池的 stat 调用数。
STAT_CHUNK = 4096


def parse_ranges(spec):
    """把 "0-99"、"5"、"0-9,20-29" 解析成整数集合。"""
    numbers = set()
    for part in str(spec).split(sep = ","):
        part = part.strip()
        if not part:
            continue
        first, dash, last = part.partition("-")
        start = int(first)
        stop = int(last) if dash else start
        if start > stop:
            raise ValueError(f"范围写反了：{part}")
        numbers.update(range(start, stop + 1))
    return numbers


def parse_steps(spec):
    """把 "30000-40000" 解析成 (30000, 40000)，表示下标 30000 到 39999 的步。"""
    first, dash, stop = str(spec).partition("-")
    if not dash or int(first) >= int(stop):
        raise ValueError(f"步数范围应写成「起-止」，止不含在内：{spec}")
    return int(first), int(stop)


def overlaps(shard, steps):
    """顺序分片的步数范围与 [起, 止) 是否相交。"""
    first, stop = steps
    return shard["first_step"] < stop and shard["last_step"] >= first


def select_shards(
    header,
    shards = None,
    pool = False,
    steps = None,
):
    """按编号范围、常驻分片、步数范围选出分片编号（取并集），排好序。"""
    chosen = set()
    if shards == "all":
        chosen.update(range(len(header["shards"])))
    elif shards:
        chosen.update(parse_ranges(spec = shards))
    covered = parse_steps(spec = steps) if steps else None
    for number, shard in enumerate(header["shards"]):
        if pool and shard["kind"] == "pool":
            chosen.add(number)
        if covered is not None and shard["kind"] == "stream" and overlaps(shard = shard, steps = covered):
            chosen.add(number)
    outside = sorted(number for number in chosen if not 0 <= number < len(header["shards"]))
    if outside:
        raise ValueError(f"分片编号超出范围 0-{len(header['shards']) - 1}：{format_ranges(numbers = outside)}")
    return sorted(chosen)


def plan_settings(run_config = None, train_args = ()):
    """训练配置：train.py 的参数解析结果，--run-config 给出时用其中的 PLAN_KEYS 各项覆盖。"""
    from pipeline.train.train import PLAN_KEYS, REQUIRED_PLACEHOLDERS, parse_args

    settings = parse_args(argv = REQUIRED_PLACEHOLDERS + list(train_args))
    if run_config:
        config = json.loads(s = Path(run_config).read_text(encoding = "utf-8"))
        for key in PLAN_KEYS:
            if key in config:
                setattr(settings, key, config[key])
    if not settings.train_manifest:
        raise SystemExit("需要 --run-config，或在 -- 之后给出 train.py 的 --train-manifest 等参数")
    return settings


def training_plan(settings):
    """用训练程序自己的函数算出训练计划，返回 (清单行, EpochPlan)。"""
    from pipeline.train.data import training_targets
    from pipeline.train.train import make_batch_stream, training_rows, training_weights

    rows = training_rows(args = settings)
    _, empty = training_targets(rows = rows, target = settings.target)
    weights = training_weights(args = settings, rows = rows, empty = empty)
    return rows, make_batch_stream(args = settings, rows = rows, weights = weights).plan


def file_sizes(paths, threads):
    """多线程取文件大小（几百万个文件时逐个 stat 很慢）。"""
    def stat_chunk(chunk):
        return [os.stat(path = path).st_size for path in chunk]

    chunks = [paths[start:start + STAT_CHUNK] for start in range(0, len(paths), STAT_CHUNK)]
    with ThreadPoolExecutor(max_workers = threads) as pool:
        return [size for sizes in pool.map(stat_chunk, chunks) for size in sizes]


def plan_groups(settings, epoch, steps):
    """按训练计划划分：返回 (分片组列表, 训练计划信息, 这一轮读到的全部音频路径)。

    每个分片组是 (种类, 文件名前缀, 音频路径列表, {路径: 读取它的步}；常驻分片为 None)。
    """
    from pipeline.train.train import PLAN_KEYS

    start = time.perf_counter()
    rows, plan = training_plan(settings = settings)
    print(f"清单 {len(rows)} 行，训练计划已算出（{time.perf_counter() - start:.0f} 秒）", flush = True)
    # 每个文件这一轮读几次、第一次在哪一步读；字典保持插入顺序，即第一次读到的顺序。
    counts = {}
    first_step = {}
    digest = hashlib.sha256()
    epoch_first = None
    epoch_last = None
    reads = 0
    for step, indices in plan_batches(plan = plan, epoch = epoch, grad_accum = settings.grad_accum):
        paths = [rows[index]["audio_path"] for index in indices]
        digest.update(digest_lines(step = step, paths = paths))
        for path in paths:
            counts[path] = counts.get(path, 0) + 1
            first_step.setdefault(path, step)
        epoch_first = step if epoch_first is None else epoch_first
        epoch_last = step
        reads += len(paths)
    covered = parse_steps(spec = steps) if steps else (epoch_first, epoch_last + 1)
    if steps:
        # 只覆盖一段步数：常驻分片只放这段步数中读到的多次读取文件。
        pool_paths = set()
        for step, indices in plan_batches(plan = plan, epoch = epoch, grad_accum = settings.grad_accum):
            if covered[0] <= step < covered[1]:
                pool_paths.update(
                    rows[index]["audio_path"] for index in indices if counts[rows[index]["audio_path"]] > 1
                )
        pool_paths = sorted(pool_paths)
        pool_prefix = f"pool-{covered[0]}-{covered[1]}"
    else:
        pool_paths = sorted(path for path, count in counts.items() if count > 1)
        pool_prefix = "pool"
    stream_paths = [path for path, count in counts.items() if count == 1]
    unread = len({row["audio_path"] for row in rows}) - len(counts)
    print(
        f"这一轮读取 {reads} 次、{len(counts)} 个文件：常驻 {len(pool_paths)} 个，顺序 {len(stream_paths)} 个；"
        f"清单中这一轮不读的文件 {unread} 个",
        flush = True,
    )
    info = {
        "epoch": epoch,
        "params": {key: getattr(settings, key) for key in PLAN_KEYS},
        "digest": digest.hexdigest(),
        "epoch_steps": [epoch_first, epoch_last + 1],
        "steps": list(covered),
        "rows": len(rows),
        "reads": reads,
    }
    groups = [("pool", pool_prefix, pool_paths, None), ("stream", "stream", stream_paths, first_step)]
    return groups, info, list(counts)


def build_index(
    settings,
    out_dir,
    name,
    order = "plan",
    epoch = 0,
    steps = None,
    shard_bytes = 10 ** 9,
    member_root = None,
    threads = 16,
):
    """划分分片并写出索引，返回索引 json 的路径。

    order 为 plan 时按训练计划分成常驻分片和顺序分片（plan_groups）；为 path 时清单（筛选后）中的全部音频按路径
    排序装入分片，与训练计划无关，steps 不能用。
    """
    from pipeline.train.train import training_rows

    start = time.perf_counter()
    source = {key: getattr(settings, key) for key in ("train_manifest", "audio_root", "langs", "max_duration_s")}
    if order == "plan":
        groups, plan, every = plan_groups(settings = settings, epoch = epoch, steps = steps)
    elif order == "path":
        if steps:
            raise ValueError("按路径排列的索引不分步数，不能用 --steps")
        rows = training_rows(args = settings)
        groups = [("path", "path", sorted({row["audio_path"] for row in rows}), None)]
        plan = None
        every = groups[0][2]
        print(f"清单 {len(rows)} 行，{len(groups[0][2])} 个音频文件", flush = True)
    else:
        raise ValueError(f"未知的排列方式：{order}")
    all_paths = [path for _, _, paths, _ in groups for path in paths]
    # 成员名的基准目录按这一轮读到的全部文件算，只覆盖一段步数的索引与完整索引相同。
    root = member_root or os.path.commonpath(paths = [os.path.dirname(path) for path in every])
    # tar 成员的修改时间取训练清单的修改时间：同一份清单生成的完整索引和只覆盖一段步数的索引，顺序分片逐字节相同。
    mtime = int(os.stat(path = settings.train_manifest).st_mtime)
    shards = []
    placed = []
    for kind, prefix, paths, steps_of in groups:
        sizes = file_sizes(paths = paths, threads = threads)
        locations, group_shards = layout(
            members = [(member_name(path = path, root = root), size) for path, size in zip(paths, sizes)],
            shard_bytes = shard_bytes,
            mtime = mtime,
        )
        first = len(shards)
        for number, (count, total) in enumerate(group_shards):
            shards.append({"name": f"{prefix}-{number:05d}.tar", "kind": kind, "files": count, "bytes": total})
        if steps_of is not None:
            for path, (number, _) in zip(paths, locations):
                shard = shards[first + number]
                shard.setdefault("first_step", steps_of[path])
                shard["last_step"] = steps_of[path]
        placed.append((paths, sizes, locations, first))

    def entries():
        for paths, sizes, locations, first in placed:
            for path, size, (number, offset) in zip(paths, sizes, locations):
                yield path, first + number, offset, size

    kinds = {}
    for shard in shards:
        entry = kinds.setdefault(shard["kind"], {"shards": 0, "files": 0, "bytes": 0})
        entry["shards"] += 1
        entry["files"] += shard["files"]
        entry["bytes"] += shard["bytes"]
    header = {
        "order": order,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "member_root": str(root),
        "member_mtime": mtime,
        "shard_bytes": int(shard_bytes),
        "source": source,
        "plan": plan,
        "files": len(all_paths),
        "total_bytes": sum(shard["bytes"] for shard in shards),
        "kinds": kinds,
        "shards": shards,
    }
    index = write_index(
        directory = out_dir,
        name = name,
        header = header,
        entries = entries(),
    )
    print(json.dumps(obj = {
        "index": str(index),
        "order": order,
        "steps": plan["steps"] if plan else None,
        "kinds": {kind: {**entry, "gb": round(number = entry["bytes"] / GB, ndigits = 2)} for kind, entry in kinds.items()},
        "seconds": round(number = time.perf_counter() - start, ndigits = 1),
    }, ensure_ascii = False), flush = True)
    return index


def read_file(path):
    with open(file = path, mode = "rb") as handle:
        return handle.read()


def entries_by_shard(header, numbers):
    """从索引中取出这些分片的条目：{分片编号: [(audio_path, 数据起点, 长度), ...]}，按起点排列。"""
    wanted = set(numbers)
    grouped = {number: [] for number in numbers}
    for path, number, offset, size in iter_entries(header = header):
        if number in wanted:
            grouped[number].append((path, offset, size))
    for entries in grouped.values():
        entries.sort(key = lambda entry: entry[1])
    return grouped


def pack_one(
    header,
    number,
    entries,
    target,
    threads,
):
    """把一个分片的成员依次写成 tar（先写 .tmp，核对每个成员的数据起点和整个文件的大小后改名）。

    读取用线程池提前读后面的文件，最多同时读 threads × 8 个，写入严格按索引的顺序。
    """
    shard = header["shards"][number]
    root = header["member_root"]
    mtime = header["member_mtime"]
    tmp = target.with_name(name = target.name + ".tmp")
    window = threads * 8
    with open(file = tmp, mode = "wb") as raw, ThreadPoolExecutor(max_workers = threads) as pool:
        with tarfile.open(
            fileobj = raw,
            mode = "w",
            format = TAR_FORMAT,
            encoding = TAR_ENCODING,
        ) as archive:
            pending = deque()
            upcoming = iter(entries)
            for entry in upcoming:
                pending.append((entry, pool.submit(read_file, path = entry[0])))
                if len(pending) >= window:
                    break
            while pending:
                (path, offset, size), future = pending.popleft()
                data = future.result()
                following = next(upcoming, None)
                if following is not None:
                    pending.append((following, pool.submit(read_file, path = following[0])))
                name = member_name(path = path, root = root)
                header_length = len(member_header(name = name, size = size, mtime = mtime))
                if len(data) != size or archive.offset + header_length != offset:
                    raise RuntimeError(
                        f"{path} 与索引不一致（索引中数据起点 {offset}、长度 {size}，实际 {archive.offset + header_length}、"
                        f"{len(data)}）；文件在生成索引后改变过，需要重新生成索引和全部分片"
                    )
                archive.addfile(
                    tarinfo = member_info(name = name, size = size, mtime = mtime),
                    fileobj = io.BytesIO(initial_bytes = data),
                )
        raw.flush()
        os.fsync(fd = raw.fileno())
    written = tmp.stat().st_size
    if written != shard["bytes"]:
        raise RuntimeError(f"{shard['name']} 写出 {written} 字节，索引中为 {shard['bytes']}")
    tmp.replace(target = target)
    return written


def command_index(args, train_args):
    out_dir = Path(args.out_dir)
    if (out_dir / f"{args.name}.json").exists() and not args.overwrite:
        raise SystemExit(
            f"{out_dir / (args.name + '.json')} 已存在。分片按索引划分，已生成或上传的分片依赖它；"
            "确实要重新划分时加 --overwrite，并删掉按旧索引生成的分片。"
        )
    build_index(
        settings = plan_settings(run_config = args.run_config, train_args = train_args),
        out_dir = out_dir,
        name = args.name,
        order = args.order,
        epoch = args.epoch,
        steps = args.steps,
        shard_bytes = int(args.shard_gb * GB),
        member_root = args.member_root,
        threads = args.threads,
    )


def command_pack(args, train_args):
    header = read_header(index_path = args.index)
    out_dir = Path(args.out_dir) if args.out_dir else Path(header["index_path"]).parent
    out_dir.mkdir(parents = True, exist_ok = True)
    numbers = select_shards(
        header = header,
        shards = args.shards,
        pool = args.pool,
        steps = args.steps,
    )
    if not numbers:
        raise SystemExit("没有选中分片：用 --shards、--pool 或 --steps 选择")
    todo = [number for number in numbers if complete_path(directory = out_dir, shard = header["shards"][number]) is None]
    if len(todo) < len(numbers):
        print(f"跳过 {len(numbers) - len(todo)} 个已完整的分片", flush = True)
    if not todo:
        return
    needed = sum(header["shards"][number]["bytes"] for number in todo)
    free = shutil.disk_usage(path = out_dir).free
    if needed > free:
        raise SystemExit(f"这批分片共 {needed / GB:.1f} GB，{out_dir} 只剩 {free / GB:.1f} GB；请缩小选择范围")
    grouped = entries_by_shard(header = header, numbers = todo)
    total_bytes = 0
    total_files = 0
    start = time.perf_counter()
    for number in todo:
        shard = header["shards"][number]
        shard_start = time.perf_counter()
        written = pack_one(
            header = header,
            number = number,
            entries = grouped.pop(number),
            target = out_dir / shard["name"],
            threads = args.threads,
        )
        seconds = max(time.perf_counter() - shard_start, 1e-9)
        total_bytes += written
        total_files += shard["files"]
        steps = f"，第 {shard['first_step']}–{shard['last_step']} 步" if shard["kind"] == "stream" else ""
        print(
            f"{shard['name']}{steps}：{shard['files']} 个文件，{written / GB:.3f} GB，{seconds:.1f} 秒，"
            f"{written / 1e6 / seconds:.0f} MB/s，{shard['files'] / seconds:.0f} 文件/s",
            flush = True,
        )
    seconds = max(time.perf_counter() - start, 1e-9)
    print(json.dumps(obj = {
        "shards": format_ranges(numbers = todo),
        "files": total_files,
        "gb": round(number = total_bytes / GB, ndigits = 3),
        "seconds": round(number = seconds, ndigits = 1),
        "mb_per_s": round(number = total_bytes / 1e6 / seconds, ndigits = 1),
        "files_per_s": round(number = total_files / seconds, ndigits = 1),
    }, ensure_ascii = False))


def command_status(args, train_args):
    header = read_header(index_path = args.index)
    dirs = parse_dirs(spec = args.dirs) or [Path(header["index_path"]).parent]
    anywhere = set()
    for directory in dirs:
        complete = [
            number for number, shard in enumerate(header["shards"])
            if complete_path(directory = directory, shard = shard) is not None
        ]
        anywhere.update(complete)
        print(json.dumps(obj = {
            "dir": str(directory),
            **{
                kind: sum(1 for number in complete if header["shards"][number]["kind"] == kind)
                for kind in header["kinds"]
            },
            "shards": format_ranges(numbers = complete),
            "gb": round(number = sum(header["shards"][number]["bytes"] for number in complete) / GB, ndigits = 1),
        }, ensure_ascii = False))
    missing = [number for number in range(len(header["shards"])) if number not in anywhere]
    print(json.dumps(obj = {
        "shards": len(header["shards"]),
        "order": header["order"],
        "steps": header["plan"]["steps"] if header["plan"] else None,
        "gb": {kind: round(number = entry["bytes"] / GB, ndigits = 1) for kind, entry in header["kinds"].items()},
        "missing": len(missing),
        "missing_shards": format_ranges(numbers = missing),
    }, ensure_ascii = False))


def command_list(args, train_args):
    """打印选中分片的文件名，每行一个，供 gcloud storage cp -I 等批量复制使用。"""
    header = read_header(index_path = args.index)
    numbers = select_shards(
        header = header,
        shards = args.shards,
        pool = args.pool,
        steps = args.steps,
    )
    prefix = f"{args.prefix.rstrip('/')}/" if args.prefix else ""
    for number in numbers:
        print(prefix + header["shards"][number]["name"])


def tar_members(path):
    """tar 文件的成员表 [(成员名, 数据起点, 长度)]。"""
    with tarfile.open(name = path, mode = "r:", encoding = TAR_ENCODING) as archive:
        return [(member.name, member.offset_data, member.size) for member in archive]


def command_verify(args, train_args):
    header = read_header(index_path = args.index)
    dirs = parse_dirs(spec = args.dirs) or [Path(header["index_path"]).parent]
    chosen = select_shards(
        header = header,
        shards = args.shards,
        pool = args.pool,
        steps = args.steps,
    ) if (args.shards or args.pool or args.steps) else list(range(len(header["shards"])))
    paths = {}
    for number in chosen:
        for directory in dirs:
            path = complete_path(directory = directory, shard = header["shards"][number])
            if path is not None:
                paths[number] = path
                break
    if not paths:
        raise SystemExit("给定目录中没有选中的完整分片")
    grouped = entries_by_shard(header = header, numbers = sorted(paths))
    root = header["member_root"]
    table_errors = []
    for number, path in sorted(paths.items()):
        expected = [(member_name(path = audio, root = root), offset, size) for audio, offset, size in grouped[number]]
        actual = tar_members(path = path)
        if actual != expected:
            difference = next(
                (index for index, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1]),
                min(len(actual), len(expected)),
            )
            table_errors.append(
                f"{path.name}：成员表与索引不一致，第 {difference} 项 "
                f"{actual[difference] if difference < len(actual) else None} ≠ "
                f"{expected[difference] if difference < len(expected) else None}"
            )
    mismatched = []
    unreadable = []
    checked = 0
    if args.samples > 0:
        population = [
            (audio, number, offset, size)
            for number in sorted(paths) for audio, offset, size in grouped[number]
        ]
        if args.samples < len(population):
            population = random.Random(x = args.seed).sample(population = population, k = args.samples)
        handles = {}
        try:
            for audio, number, offset, size in population:
                try:
                    original = read_file(path = audio)
                except OSError as error:
                    unreadable.append(f"{audio}: {error}")
                    continue
                if number not in handles:
                    handles[number] = open(file = paths[number], mode = "rb")
                handle = handles[number]
                handle.seek(offset)
                checked += 1
                if handle.read(size) != original:
                    mismatched.append(audio)
        finally:
            for handle in handles.values():
                handle.close()
    print(json.dumps(obj = {
        "shards_checked": format_ranges(numbers = paths),
        "table_errors": table_errors[:20],
        "bytes_checked": checked,
        "bytes_mismatched": mismatched[:20],
        "original_unreadable": unreadable[:20],
    }, ensure_ascii = False))
    if table_errors or mismatched or unreadable:
        sys.exit(1)


def parse_args(argv):
    parser = argparse.ArgumentParser(description = "按训练计划把训练音频打包成 tar 分片，供 Colab 训练读取")
    commands = parser.add_subparsers(dest = "command", required = True)

    index = commands.add_parser(name = "index", help = "按训练计划生成分片索引；训练参数写在 -- 之后")
    index.add_argument("--out-dir", required = True, help = "索引和分片所在目录")
    index.add_argument("--name", default = "index", help = "索引文件名（不含扩展名），默认 index")
    index.add_argument("--run-config", default = None, help = "已有训练的 train_config.json，用其中的训练计划配置")
    index.add_argument(
        "--order",
        choices = ("plan", "path"),
        default = "plan",
        help = "plan：按训练计划排列（Drive 加本地窗口）；path：按路径排列（随机读取的存储）",
    )
    index.add_argument(
        "--epoch",
        type = int,
        default = 0,
        help = "按第几轮（从 0 起）的计划排列",
    )
    index.add_argument("--steps", default = None, help = "只覆盖这段步数，例如 30000-40000（止不含在内）")
    index.add_argument(
        "--shard-gb",
        type = float,
        default = 1.0,
        help = "每个分片的目标大小（GB，10^9 字节）",
    )
    index.add_argument("--member-root", default = None, help = "tar 成员名的基准目录，默认是全部音频的公共上级目录")
    index.add_argument(
        "--threads",
        type = int,
        default = 16,
        help = "取文件大小的线程数",
    )
    index.add_argument("--overwrite", action = "store_true", help = "覆盖同名索引")

    pack = commands.add_parser(name = "pack", help = "按索引生成一批分片")
    pack.add_argument("--index", required = True, help = "索引 json")
    pack.add_argument("--shards", default = None, help = "分片编号范围，例如 48-147、0-9,20-29 或 all")
    pack.add_argument("--pool", action = "store_true", help = "全部常驻分片")
    pack.add_argument("--steps", default = None, help = "服务这段步数的顺序分片，例如 30000-40000")
    pack.add_argument("--out-dir", default = None, help = "分片写到哪里，默认是索引所在目录")
    pack.add_argument(
        "--threads",
        type = int,
        default = 8,
        help = "读取音频文件的线程数",
    )

    status = commands.add_parser(name = "status", help = "查看各目录中哪些分片已经完整")
    status.add_argument("--index", required = True)
    status.add_argument("--dirs", default = None, help = "分片目录，多个用冒号分隔，默认是索引所在目录")

    listing = commands.add_parser(name = "list", help = "打印选中分片的文件名")
    listing.add_argument("--index", required = True)
    listing.add_argument("--shards", default = None, help = "分片编号范围或 all")
    listing.add_argument("--pool", action = "store_true", help = "全部常驻分片")
    listing.add_argument("--steps", default = None, help = "服务这段步数的顺序分片")
    listing.add_argument("--prefix", default = None, help = "加在文件名前的目录")

    verify = commands.add_parser(name = "verify", help = "核对 tar 成员表与索引一致，抽查成员字节与原文件一致")
    verify.add_argument("--index", required = True)
    verify.add_argument("--dirs", default = None, help = "分片目录，多个用冒号分隔，默认是索引所在目录")
    verify.add_argument("--shards", default = None, help = "只核对这些分片，默认核对目录中全部完整的分片")
    verify.add_argument("--pool", action = "store_true")
    verify.add_argument("--steps", default = None)
    verify.add_argument(
        "--samples",
        type = int,
        default = 500,
        help = "与原文件逐字节比较的成员数；0 只核对成员表（原文件不在这台机器上时）",
    )
    verify.add_argument("--seed", type = int, default = 0)
    return parser.parse_args(args = argv)


COMMANDS = {
    "index": command_index,
    "pack": command_pack,
    "status": command_status,
    "list": command_list,
    "verify": command_verify,
}


def main(argv = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    train_args = []
    if "--" in argv:
        split = argv.index("--")
        argv, train_args = argv[:split], argv[split + 1:]
    args = parse_args(argv = argv)
    COMMANDS[args.command](args = args, train_args = train_args)


if __name__ == "__main__":
    main()
