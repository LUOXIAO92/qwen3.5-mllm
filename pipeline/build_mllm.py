"""从一个 Qwen3.5 backbone 构建融合模型目录（在 envs/mllm-eval 中运行，任务 #60）。

阶段 1 的教材 01-module-preparation-zh.ipynb 用 Qwen3.5-0.8B 一步步做了融合；这里把同样的步骤整理成脚本，
用于构建 2B、4B（用户 2026-11-15 要求 v2b 同时训练三个尺寸）。步骤与教材相同：

1. 从 --code-from（如 qwen3.5-0.8b-mllm）复制自定义代码文件，融合模型目录随这些 .py 一起保存，可以用 remote code 加载；
2. 用 backbone 的配置加上音频塔的配置生成融合配置；
3. 用 backbone 的 processor 加上音频特征提取器生成融合 processor，聊天模板加入音频分支，
   分词器登记 <|audio_start|>、<|audio_end|>，配置记下 <|audio_pad|> 与起止 token 的编号；
4. 视觉塔、语言模型与输出层取自 backbone，音频塔取自 --audio-tower，projector（Linear → SiLU → Linear）按 --seed 新初始化，
   输出维度等于 backbone 语言模型的 hidden size。权重默认以 bfloat16 保存（--save-dtype）：backbone 与音频塔的原始
   权重本来就是 bf16，存成 float32 只会让文件翻倍；训练时冻结部分按 bf16 加载、projector 转成 float32（pipeline/train/model.py）。
   阶段 1 教材把 0.8B 存成了 float32，用 --save-dtype float32 可以复现。

分词器、模板和视觉塔都按 backbone 各自的文件生成：三个尺寸的词表相同，但 4B 的聊天模板与 tokenizer_config 不同于
0.8B、2B，视觉塔的尺寸也不同，不能直接复制 0.8B 的。

用法（在仓库根目录运行）::

    python -m pipeline.build_mllm --backbone <weights/Qwen3.5-2B> --audio-tower <weights/qwen3-omni-audio> \\
        --code-from qwen3.5-0.8b-mllm --out qwen3.5-2b-mllm
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

CODE_FILES = [
    "configuration_qwen3_5_mllm.py",
    "modeling_qwen3_5_mllm.py",
    "processing_qwen3_5_mllm.py",
    "configuration_qwen3_omni_audio.py",
    "modeling_qwen3_omni_audio.py",
    "audio_video_utils.py",
    "media_io.py",
]
AUDIO_TEMPLATE_BRANCH = """{%- elif item.type == 'audio' %}
                {{- '<|audio_start|><|audio_pad|><|audio_end|>' }}
            """
TEXT_TEMPLATE_BRANCH = "{%- elif 'text' in item %}"


def build(backbone: Path, audio_tower: Path, code_from: Path, out: Path, seed: int, save_dtype: torch.dtype) -> None:
    from transformers import AutoConfig, AutoFeatureExtractor, AutoProcessor, Qwen3_5ForConditionalGeneration
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    out.mkdir(parents = True, exist_ok = True)
    for name in CODE_FILES:
        shutil.copy2(src = code_from / name, dst = out / name)

    def dynamic(reference: str):
        return get_class_from_dynamic_module(class_reference = reference, pretrained_model_name_or_path = out)

    config_class = dynamic(reference = "configuration_qwen3_5_mllm.Qwen3_5MLLMConfig")
    processor_class = dynamic(reference = "processing_qwen3_5_mllm.Qwen3_5MLLMProcessor")
    model_class = dynamic(reference = "modeling_qwen3_5_mllm.Qwen3_5MLLMForConditionalGeneration")
    projector_class = dynamic(reference = "modeling_qwen3_5_mllm.Qwen3_5MLLMAudioProjector")
    audio_class = dynamic(reference = "modeling_qwen3_omni_audio.Qwen3OmniAudioTowerModel")

    # 融合配置：backbone 的配置加上音频塔的配置
    backbone_config = AutoConfig.from_pretrained(pretrained_model_name_or_path = backbone)
    backbone_config_dict = backbone_config.to_dict()
    backbone_config_dict.pop("model_type", None)
    backbone_config_dict.pop("architectures", None)
    audio_tower_config = AutoConfig.from_pretrained(pretrained_model_name_or_path = audio_tower, trust_remote_code = True)
    mllm_config = config_class(audio_config = audio_tower_config.audio_config.to_dict(), **backbone_config_dict)

    # 融合 processor：聊天模板加入音频分支，分词器登记音频起止 token
    audio_processor = AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path = audio_tower)
    backbone_processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path = backbone)
    if TEXT_TEMPLATE_BRANCH not in backbone_processor.chat_template:
        raise SystemExit(f"{backbone} 的聊天模板里没有 {TEXT_TEMPLATE_BRANCH!r}，无法插入音频分支")
    chat_template = backbone_processor.chat_template.replace(
        TEXT_TEMPLATE_BRANCH,
        AUDIO_TEMPLATE_BRANCH + TEXT_TEMPLATE_BRANCH,
        1,
    )
    tokenizer = backbone_processor.tokenizer
    tokenizer.add_special_tokens(special_tokens_dict = {"audio_bos_token": "<|audio_start|>", "audio_eos_token": "<|audio_end|>"})
    processor = processor_class(
        image_processor = backbone_processor.image_processor,
        tokenizer = tokenizer,
        video_processor = backbone_processor.video_processor,
        feature_extractor = audio_processor,
        chat_template = chat_template,
        audio_n_window = mllm_config.audio_config.n_window,
    )
    mllm_config.audio_token_id = tokenizer.convert_tokens_to_ids(tokens = "<|audio_pad|>")
    mllm_config.audio_start_token_id = tokenizer.audio_bos_token_id
    mllm_config.audio_end_token_id = tokenizer.audio_eos_token_id

    # 完整模型：视觉塔、语言模型、输出层取自 backbone，音频塔取自导出的 Qwen3-Omni 音频塔，projector 新初始化
    backbone_with_head = Qwen3_5ForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path = backbone,
        dtype = torch.float32,
        attn_implementation = "eager",
    )
    audio_component = audio_class.from_pretrained(
        pretrained_model_name_or_path = audio_tower,
        dtype = torch.float32,
        attn_implementation = "eager",
    )
    torch.manual_seed(seed = seed)
    projector = projector_class(
        audio_output_dim = mllm_config.audio_config.output_dim,
        text_hidden_size = mllm_config.text_config.hidden_size,
    )
    full_model = model_class(config = mllm_config)
    full_model.model.visual = backbone_with_head.model.visual
    full_model.model.language_model = backbone_with_head.model.language_model
    full_model.lm_head = backbone_with_head.lm_head
    full_model.model.audio_tower = audio_component.model.audio_tower
    full_model.model.audio_projector = projector
    full_model.tie_weights()
    full_model.eval()

    config_class.register_for_auto_class(auto_class = "AutoConfig")
    processor_class.register_for_auto_class(auto_class = "AutoProcessor")
    model_class.register_for_auto_class(auto_class = "AutoModelForImageTextToText")
    # 与 0.8B 相同的入口：配置、基础模型和带输出层的完整模型都从目录里的自定义代码加载
    full_model.config.auto_map = {
        "AutoConfig": "configuration_qwen3_5_mllm.Qwen3_5MLLMConfig",
        "AutoModel": "modeling_qwen3_5_mllm.Qwen3_5MLLMModel",
        "AutoModelForImageTextToText": "modeling_qwen3_5_mllm.Qwen3_5MLLMForConditionalGeneration",
    }
    # 权重按 save_dtype 保存，顶层与各子配置也记为同一精度（backbone 的 text_config 原本记的是 bfloat16）
    full_model.to(dtype = save_dtype)
    full_model.config.dtype = save_dtype
    for sub_config in (full_model.config.text_config, full_model.config.vision_config, full_model.config.audio_config):
        sub_config.dtype = save_dtype
    full_model.save_pretrained(save_directory = out)
    processor.save_pretrained(save_directory = out)


def verify(out: Path) -> dict:
    """重新加载融合模型目录，用一条 2 秒的合成音频跑一次前向和 generate，返回关键编号与形状。"""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        pretrained_model_name_or_path = out,
        trust_remote_code = True,
        dtype = torch.float32,
        attn_implementation = "eager",
    ).eval()
    processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path = out, trust_remote_code = True)
    sampling_rate = processor.feature_extractor.sampling_rate
    times = np.arange(2 * sampling_rate) / sampling_rate
    wave = (0.1 * np.sin(2 * np.pi * 440 * times)).astype(np.float32)
    messages = [{"role": "user", "content": [{"type": "audio", "audio": "tone"}, {"type": "text", "text": "Transcribe the audio."}]}]
    prompt = processor.apply_chat_template(
        conversation = messages,
        tokenize = False,
        add_generation_prompt = True,
        enable_thinking = False,
    )
    inputs = processor(text = prompt, audio = [wave], sampling_rate = sampling_rate, return_tensors = "pt")
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens = 4,
            do_sample = False,
            pad_token_id = processor.tokenizer.pad_token_id,
        )
    config = model.config
    return {
        "text_hidden_size": config.text_config.hidden_size,
        "text_layers": config.text_config.num_hidden_layers,
        "projector": [tuple(parameter.shape) for parameter in model.model.audio_projector.parameters()],
        "audio_token_id": config.audio_token_id,
        "audio_start_token_id": config.audio_start_token_id,
        "audio_end_token_id": config.audio_end_token_id,
        "audio_tokens_in_prompt": int((inputs["input_ids"] == config.audio_token_id).sum()),
        "generated_tokens": int(generated.shape[1] - inputs["input_ids"].shape[1]),
        "parameters_billion": round(sum(parameter.numel() for parameter in model.parameters()) / 1e9, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description = "从 Qwen3.5 backbone 构建融合模型目录")
    parser.add_argument("--backbone", required = True, type = Path)
    parser.add_argument("--audio-tower", required = True, type = Path)
    parser.add_argument("--code-from", required = True, type = Path, help = "提供自定义代码文件的融合模型目录（如 qwen3.5-0.8b-mllm）")
    parser.add_argument("--out", required = True, type = Path)
    parser.add_argument("--seed", type = int, default = 0, help = "projector 初始化的随机种子")
    parser.add_argument("--save-dtype", choices = ["bfloat16", "float32"], default = "bfloat16", help = "权重的保存精度")
    parser.add_argument("--verify-only", action = "store_true", help = "只重新加载并核对已构建的目录")
    args = parser.parse_args()
    if not args.verify_only:
        build(
            backbone = args.backbone,
            audio_tower = args.audio_tower,
            code_from = args.code_from,
            out = args.out,
            seed = args.seed,
            save_dtype = getattr(torch, args.save_dtype),
        )
    print(json.dumps(obj = verify(out = args.out), ensure_ascii = False))


if __name__ == "__main__":
    main()
