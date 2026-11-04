"""projector 检查点的保存、读取，以及把 projector 合入阶段 1 模型导出完整模型目录。

检查点目录 <output-dir>/checkpoint-<步数>/ 中有三个文件：

- projector.safetensors：projector 权重，只含完整模型 state_dict 中以 model.audio_projector.
  开头的四个键，键名与完整模型一致，float32。
- projector.json：训练配置、步数、训练目标、最近的训练损失与 dev 损失。
- trainer_state.pt：断点续训用的优化器、学习率调度器、随机数状态和数据位置。默认只在最近的
  检查点中保留，较早检查点只保留前两个文件。

<output-dir>/latest 记录最近一个完整写出的检查点目录名。
"""

import json
import shutil
from pathlib import Path

import torch

from pipeline.train.model import PROJECTOR_PREFIX, projector_state_dict

PROJECTOR_FILE = "projector.safetensors"
INFO_FILE = "projector.json"
STATE_FILE = "trainer_state.pt"
LATEST_FILE = "latest"
FORMAT_NAME = "qwen3.5-mllm-audio-projector/1"


def checkpoint_name(step):
    return f"checkpoint-{step:07d}"


def save_checkpoint(
    output_dir,
    step,
    model,
    info,
    trainer_state,
    keep_trainer_states = 1,
):
    """写出一个检查点。先写到临时目录再改名，中途中断不会留下不完整的检查点。"""
    from safetensors.torch import save_file

    output_dir = Path(output_dir)
    directory = output_dir / checkpoint_name(step = step)
    staging = output_dir / (directory.name + ".tmp")
    if staging.exists():
        shutil.rmtree(path = staging)
    staging.mkdir(parents = True)

    state = projector_state_dict(model = model)
    save_file(
        tensors = state,
        filename = str(staging / PROJECTOR_FILE),
        metadata = {"format": "pt"},
    )
    info = {
        "format": FORMAT_NAME,
        "step": step,
        "projector_keys": sorted(state),
        "dtype": "float32",
        **info,
    }
    (staging / INFO_FILE).write_text(
        data = json.dumps(obj = info, ensure_ascii = False, indent = 2) + "\n",
        encoding = "utf-8",
    )
    torch.save(obj = trainer_state, f = staging / STATE_FILE)

    if directory.exists():
        shutil.rmtree(path = directory)
    staging.rename(target = directory)
    (output_dir / LATEST_FILE).write_text(data = directory.name + "\n", encoding = "utf-8")

    if keep_trainer_states > 0:
        with_state = sorted(
            path for path in output_dir.glob(pattern = "checkpoint-*")
            if (path / STATE_FILE).is_file()
        )
        for path in with_state[:-keep_trainer_states]:
            (path / STATE_FILE).unlink()
    return directory


def latest_checkpoint(output_dir):
    """返回最近一个可用于续训的检查点目录，没有时返回 None。"""
    output_dir = Path(output_dir)
    latest = output_dir / LATEST_FILE
    if latest.is_file():
        directory = output_dir / latest.read_text(encoding = "utf-8").strip()
        if (directory / STATE_FILE).is_file():
            return directory
    candidates = sorted(
        path for path in output_dir.glob(pattern = "checkpoint-*")
        if path.is_dir() and (path / STATE_FILE).is_file()
    )
    return candidates[-1] if candidates else None


def read_projector(path):
    """读取 projector 权重。path 可以是 safetensors 文件，也可以是检查点目录。"""
    from safetensors.torch import load_file

    path = Path(path)
    if path.is_dir():
        path = path / PROJECTOR_FILE
    state = load_file(filename = str(path), device = "cpu")
    outside = [name for name in state if not name.startswith(PROJECTOR_PREFIX)]
    if outside:
        raise ValueError(f"{path} 含有 projector 以外的键：{outside[:5]}")
    return state


def _weight_files(model_dir):
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(s = index_path.read_text(encoding = "utf-8"))
        return sorted(set(index["weight_map"].values()))
    return ["model.safetensors"]


def export_full_model(model_dir, projector_path, output_dir):
    """把 projector 权重合入阶段 1 模型，导出一个可以直接加载的完整模型目录。

    除权重文件以外的文件（配置、remote code、processor、chat template）原样复制；
    权重文件中只替换 projector 的四个张量，其余张量按原精度原样写回，不经过模型加载。
    返回逐张量核对的结果。
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    if output_dir.resolve() == model_dir.resolve():
        raise ValueError("导出目录不能是阶段 1 模型目录本身。")
    projector = read_projector(path = projector_path)
    output_dir.mkdir(parents = True, exist_ok = True)

    weight_files = _weight_files(model_dir = model_dir)
    for path in model_dir.iterdir():
        if path.is_file() and path.name not in weight_files:
            shutil.copy2(src = path, dst = output_dir / path.name)

    replaced = set()
    for file_name in weight_files:
        source = model_dir / file_name
        with safe_open(filename = str(source), framework = "pt", device = "cpu") as handle:
            names = list(handle.keys())
            if not any(name in projector for name in names):
                shutil.copy2(src = source, dst = output_dir / file_name)
                continue
            metadata = handle.metadata()
            tensors = {}
            for name in names:
                tensor = handle.get_tensor(name = name)
                if name in projector:
                    new_tensor = projector[name]
                    if tuple(new_tensor.shape) != tuple(tensor.shape):
                        raise ValueError(f"{name} 形状为 {tuple(new_tensor.shape)}，模型中为 {tuple(tensor.shape)}")
                    tensor = new_tensor.to(dtype = tensor.dtype).contiguous()
                    replaced.add(name)
                tensors[name] = tensor
        save_file(
            tensors = tensors,
            filename = str(output_dir / file_name),
            metadata = metadata,
        )
    missing = sorted(set(projector) - replaced)
    if missing:
        raise ValueError(f"模型权重中找不到这些 projector 键：{missing}")

    info_path = Path(projector_path)
    info_path = info_path / INFO_FILE if info_path.is_dir() else info_path.with_name(name = INFO_FILE)
    if info_path.is_file():
        shutil.copy2(src = info_path, dst = output_dir / INFO_FILE)
    return verify_export(model_dir = model_dir, output_dir = output_dir, projector = projector)


def verify_export(model_dir, output_dir, projector):
    """逐张量比较导出目录与阶段 1 模型：projector 等于检查点，其余张量逐位相同。"""
    from safetensors import safe_open

    counts = {"projector_equal": 0, "other_equal": 0, "other_changed": [], "projector_mismatch": []}
    for file_name in _weight_files(model_dir = Path(model_dir)):
        with safe_open(filename = str(Path(model_dir) / file_name), framework = "pt", device = "cpu") as before, \
                safe_open(filename = str(Path(output_dir) / file_name), framework = "pt", device = "cpu") as after:
            if set(before.keys()) != set(after.keys()):
                raise ValueError(f"{file_name} 导出前后的键不一致。")
            for name in before.keys():
                new_tensor = after.get_tensor(name = name)
                if name in projector:
                    expected = projector[name].to(dtype = new_tensor.dtype)
                    if torch.equal(input = new_tensor, other = expected):
                        counts["projector_equal"] += 1
                    else:
                        counts["projector_mismatch"].append(name)
                elif torch.equal(input = new_tensor, other = before.get_tensor(name = name)):
                    counts["other_equal"] += 1
                else:
                    counts["other_changed"].append(name)
    return counts
