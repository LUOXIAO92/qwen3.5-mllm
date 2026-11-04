"""把 projector 检查点合入阶段 1 模型，导出完整模型目录。

用法：

    python -m pipeline.train.export \\
        --model-dir <阶段 1 模型目录> --projector <检查点目录或 projector.safetensors> --output-dir <导出目录>

导出目录可以直接用 AutoModelForImageTextToText.from_pretrained(..., trust_remote_code = True) 加载。
导出后逐张量核对：projector 的四个张量等于检查点，其余张量与阶段 1 模型逐位相同。
"""

import argparse
import json

from pipeline.train.checkpoint import export_full_model


def main(argv = None):
    parser = argparse.ArgumentParser(description = "projector 合入阶段 1 模型并导出")
    parser.add_argument("--model-dir", required = True)
    parser.add_argument("--projector", required = True)
    parser.add_argument("--output-dir", required = True)
    args = parser.parse_args(args = argv)

    result = export_full_model(
        model_dir = args.model_dir,
        projector_path = args.projector,
        output_dir = args.output_dir,
    )
    print(json.dumps(obj = result, ensure_ascii = False))
    if result["other_changed"] or result["projector_mismatch"]:
        raise SystemExit("导出核对未通过。")
    print(f"已导出到 {args.output_dir}：projector {result['projector_equal']} 个张量已替换，"
          f"其余 {result['other_equal']} 个张量与原模型逐位相同。")


if __name__ == "__main__":
    main()
