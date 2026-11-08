"""找出一次训练中 dev 损失最低的检查点（实验 v1，任务 T16）。

每个检查点目录的 projector.json 记录了保存时最近一次 dev 评估的结果。只有评估步数与检查点步数相同的检查点
才参与比较，按各语言损失的平均（lang_mean_loss）取最低者。

用法::

    python -m pipeline.train.best_checkpoint <训练输出目录>

打印检查点目录的路径；找不到带 dev 损失的检查点时以非零状态退出。
"""

import json
import sys
from pathlib import Path

from .checkpoint import INFO_FILE


def best_checkpoint(output_dir: Path):
    """返回 (dev 损失, 检查点目录)，没有可比较的检查点时返回 None。"""
    best = None
    for directory in sorted(Path(output_dir).glob(pattern = "checkpoint-*")):
        info = json.loads(s = (directory / INFO_FILE).read_text(encoding = "utf-8"))
        evaluation = info.get("eval") or {}
        if evaluation.get("step") != int(directory.name.split("-")[1]):
            continue
        loss = evaluation["overall"]["lang_mean_loss"]
        if best is None or loss < best[0]:
            best = (loss, directory)
    return best


def main() -> None:
    best = best_checkpoint(output_dir = Path(sys.argv[1]))
    if best is None:
        sys.exit("没有带 dev 损失的检查点")
    print(best[1])


if __name__ == "__main__":
    main()
