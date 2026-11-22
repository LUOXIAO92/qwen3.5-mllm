"""把阶段 1 融合模型里的语言模型导出成 vLLM 可以直接加载的目录，作为回放文本的参考模型。

    python -m pipeline.replay.export_lm --mllm <qwen3.5-2b-mllm 目录> --original <Qwen3.5-2B 原始目录> --out <输出目录>

参考模型是未经训练的融合模型中的语言模型。它与原始 Qwen3.5-2B 的文本权重只有一处差别：融合模型把
线性注意力层的 `A_log` 和 `norm.weight`（原始检查点中为 float32）存成了 bfloat16，数值等于原值舍入到
bfloat16。ASR 训练从融合模型出发，所以回放文本和它的对数概率都用融合模型里的这一份权重生成。

输出目录包含：融合模型中 `model.language_model.*` 的全部张量（`model.safetensors`），以及原始目录的
`config.json`、分词器文件和 chat template。原始目录的 tokenizer_config.json 写的分词器类是 Qwen2Tokenizer，
transformers 和 vLLM 按这个类加载时把泰语、阿拉伯语等的组合用符号单独切开（#77），复制后用 `pipeline.tokenizer_fix`
改为 Qwen3_5Tokenizer。vLLM 按原始架构 `Qwen3_5ForConditionalGeneration` 加载，开启 `language_model_only` 后
不建视觉塔，因此不需要视觉权重；原始检查点中的 MTP 层也不导出。
"""

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from pipeline.tokenizer_fix import fix_model_dir

PREFIX = "model.language_model."
# 原始目录中需要复制的小文件：模型配置、分词器、chat template、图像与视频预处理配置（vLLM 构建处理器时读取）
COPY_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)


def export(mllm_dir, original_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents = True, exist_ok = True)
    tensors = {}
    with safe_open(filename = str(Path(mllm_dir) / "model.safetensors"), framework = "pt", device = "cpu") as handle:
        for name in handle.keys():
            if name.startswith(PREFIX):
                tensors[name] = handle.get_tensor(name)
    save_file(tensors = tensors, filename = str(out_dir / "model.safetensors"), metadata = {"format": "pt"})
    for name in COPY_FILES:
        source = Path(original_dir) / name
        if source.exists():
            shutil.copy2(src = source, dst = out_dir / name)
    fix_model_dir(model_dir = out_dir)
    # 生成配置：只声明结束符，采样参数一律由调用方显式给出
    generation_config = {"eos_token_id": [248044, 248046]}
    (out_dir / "generation_config.json").write_text(json.dumps(generation_config, indent = 2) + "\n", encoding = "utf-8")
    total = sum(tensor.numel() for tensor in tensors.values())
    print(f"导出 {len(tensors)} 个张量，共 {total / 1e9:.3f}B 参数 -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mllm", required = True, help = "阶段 1 融合模型目录（含 model.safetensors）")
    parser.add_argument("--original", required = True, help = "原始 Qwen3.5-2B 目录（取 config.json 与分词器文件）")
    parser.add_argument("--out", required = True, help = "输出目录")
    args = parser.parse_args()
    export(mllm_dir = args.mllm, original_dir = args.original, out_dir = args.out)


if __name__ == "__main__":
    main()
