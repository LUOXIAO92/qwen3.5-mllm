"""把阶段 1 融合模型里的语言模型导出成 vLLM 可以直接加载的目录，作为回放文本的参考模型；也可以换上全量微调后的语言模型。

    python -m pipeline.replay.export_lm --mllm <qwen3.5-2b-mllm 目录> --original <Qwen3.5-2B 原始目录> --out <输出目录>
    python -m pipeline.replay.export_lm --mllm <qwen3.5-2b-mllm 目录> --original <已导出的参考模型目录> \\
        --language-model <检查点>/language_model.safetensors --out <输出目录>

参考模型是未经训练的融合模型中的语言模型。它与原始 Qwen3.5-2B 的文本权重只有一处差别：融合模型把
线性注意力层的 `A_log` 和 `norm.weight`（原始检查点中为 float32）存成了 bfloat16，数值等于原值舍入到
bfloat16。ASR 训练从融合模型出发，所以回放文本和它的对数概率都用融合模型里的这一份权重生成。

给出 --language-model（全量微调检查点中的 language_model.safetensors，#71 的文本基准用它在 vLLM 上评测微调后的模型）时，
融合模型中的语言模型张量逐个换成检查点中的同名张量。检查点的键必须与融合模型的 `model.language_model.*` 完全相同、
形状一致，数据类型按融合模型中的转换，与 transformers 评测时换权重（pipeline/eval/run_eval.py 的 load_language_model）
的要求和结果相同。这时 --original 可以直接给已经导出的参考模型目录，配置和分词器文件从那里复制。

输出目录包含：语言模型的全部张量（`model.safetensors`），以及 --original 目录的 `config.json`、分词器文件和
chat template。原始目录的 tokenizer_config.json 写的分词器类是 Qwen2Tokenizer，transformers 和 vLLM 按这个类加载时
把泰语、阿拉伯语等的组合用符号单独切开（#77），复制后用 `pipeline.tokenizer_fix` 改为 Qwen3_5Tokenizer（已经改好时不动）。
vLLM 按原始架构 `Qwen3_5ForConditionalGeneration` 加载，开启 `language_model_only` 后不建视觉塔，因此不需要视觉权重；
原始检查点中的 MTP 层也不导出。
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


def language_model_tensors(mllm_dir, language_model = None):
    """融合模型中 `model.language_model.*` 的全部张量，给出 language_model 时换成检查点中的同名张量。

    返回 (张量字典, 换成检查点的张量个数)。检查点的键与融合模型不一致、形状不同时报错；换上的张量转成融合模型中的
    数据类型。先读检查点的文件头核对，再逐个读张量，内存中只保留一份语言模型。
    """
    with safe_open(filename = str(Path(mllm_dir) / "model.safetensors"), framework = "pt", device = "cpu") as base:
        names = [name for name in base.keys() if name.startswith(PREFIX)]
        if language_model is None:
            return {name: base.get_tensor(name) for name in names}, 0
        with safe_open(filename = str(language_model), framework = "pt", device = "cpu") as tuned:
            tuned_names = set(tuned.keys())
            missing = sorted(set(names) - tuned_names)
            extra = sorted(tuned_names - set(names))
            if missing or extra:
                raise ValueError(f"{language_model} 的键与融合模型的语言模型不一致。缺少：{missing[:5]}（共 {len(missing)} 个）；多余：{extra[:5]}（共 {len(extra)} 个）")
            for name in names:
                base_shape = list(base.get_slice(name).get_shape())
                tuned_shape = list(tuned.get_slice(name).get_shape())
                if base_shape != tuned_shape:
                    raise ValueError(f"{name} 形状不一致：检查点 {tuned_shape}，融合模型 {base_shape}")
            tensors = {}
            for name in names:
                dtype = torch_dtype(label = base.get_slice(name).get_dtype())
                tensor = tuned.get_tensor(name)
                tensors[name] = tensor if tensor.dtype == dtype else tensor.to(dtype = dtype)
    return tensors, len(tensors)


def torch_dtype(label):
    """safetensors 文件头中数据类型的写法（BF16、F32 等）对应的 torch 数据类型。"""
    import torch

    table = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64}
    if label not in table:
        raise ValueError(f"数据类型 {label} 不在预期之内")
    return table[label]


def export(mllm_dir, original_dir, out_dir, language_model = None):
    """导出语言模型目录；返回导出的张量个数与其中换成检查点的个数。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents = True, exist_ok = True)
    tensors, replaced = language_model_tensors(mllm_dir = mllm_dir, language_model = language_model)
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
    source = "融合模型" if language_model is None else f"{language_model}（换掉 {replaced} 个）"
    print(f"导出 {len(tensors)} 个张量，共 {total / 1e9:.3f}B 参数，来自{source} -> {out_dir}", flush = True)
    return {"tensors": len(tensors), "replaced": replaced}


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mllm", required = True, help = "阶段 1 融合模型目录（含 model.safetensors）")
    parser.add_argument("--original", required = True, help = "原始 Qwen3.5-2B 目录或已导出的参考模型目录（取 config.json 与分词器文件）")
    parser.add_argument("--language-model", default = None, help = "全量微调检查点中的 language_model.safetensors；不给时导出融合模型中的语言模型")
    parser.add_argument("--out", required = True, help = "输出目录")
    args = parser.parse_args()
    export(
        mllm_dir = args.mllm,
        original_dir = args.original,
        out_dir = args.out,
        language_model = args.language_model,
    )


if __name__ == "__main__":
    main()
