"""VieNeu-TTS-v3-Turbo 封装（越南语，内置音色与克隆）。

模型：pnnbao-ump/VieNeu-TTS-v3-Turbo（SDK `vieneu` 3.7.1），原生输出 48 kHz，
本封装转为 16 kHz 单声道 FLAC。

支持的语言代码：只有 vi（模型卡另说明支持 en 与 vi 的混合文本，本项目只用 vi）。
其他语言的任务结果 status 为 unsupported_lang。

音色：23 个内置音色，名称以 `vieneu.list_preset_voices()` 为准，完整表见 GitHub issue #12。
克隆：支持。mode 为 clone 时，任务里用 ref_audio 给出 3 到 8 秒的参考音频路径，不需要参考文本。
内置音色模式（mode 为 builtin）用 builtin_name（其次 voice_id）给出音色名。

环境：envs/tts-vieneu（uv，Python 3.12，torch 2.8.0+cu128，transformers 4.57.6，vieneu 3.7.1）。
有 CUDA 时 SDK 自动使用 PyTorch 后端并批量生成，所以需要用 flock 占用 GPU：
    flock /home/luoxiao/workspace/qwen3.5-asr-work/gpu.lock \
        python -m pipeline.tts.vieneu --input jobs.jsonl --output-dir out --batch-size 16
显存与吞吐见 GitHub issue #12 的验收评论。

许可：模型、代码、内置音色资源均为 Apache-2.0（模型卡明确内置音色的说话人已授权用于 AI 训练与合成语音）。
克隆他人声音需要对方同意；本项目的克隆只使用本项目内置音色合成的参考片段。
音频水印：SDK 在装了 resemble-perth 时才会加水印，本环境没有安装，调用时也显式传 apply_watermark = False。

文本处理：VieNeu 的输入是音素，SDK 默认先用 sea-g2p 的 Normalizer 规范化文本（数字读法、符号、括号等），
再做 G2P。朗读稿已经由本项目完成规范化，因此本封装关闭 Normalizer，只保留 G2P（见 `_关闭文本规范化`），
以及 SDK 固定保留的断句与句末标点补全（句子过短时把句末标点统一为句号，句子过长且没有句末标点时补句号，
只涉及标点）。方括号里的 [cười] 等情绪标签是 SDK 的功能，朗读稿里不会出现。

输入每行（JSONL）：sample_id、lang、text、mode（缺省 builtin）、builtin_name 或 voice_id、可选 seed、
克隆时的 ref_audio。其余字段原样带入结果。
输出：<output-dir>/<sample_id>.flac，以及 results.jsonl（可用 --results 改路径）；
结果多出 audio_path、duration_s、status（ok 或 failed、unsupported_lang）、error。
同一个音色的任务放进同一批，种子按批设定（同一批内同一顺序可复现）。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

目标采样率 = 16000


def _关闭文本规范化():
    """关闭 sea-g2p 的文本规范化，只保留 G2P。

    SDK 分句时调用 `_get_normalizer()` 取得规范化器；换成原样返回的规范化器。
    音素化时 `_phonemize_cached` 原本会先规范化再 G2P；换成只做 G2P。
    """
    import vieneu_utils.phonemize_text as 模块

    class 原样规范化器:
        def normalize(self, text, punc_norm = True):
            return text

        def normalize_batch(self, texts, punc_norm = True):
            return list(texts)

    模块._get_normalizer = lambda: 原样规范化器()
    模块._phonemize_cached.cache_clear()

    def 只做音素化(text, punc_norm = True):
        return 模块._get_g2p().phonemize_batch([text], punc_norm = punc_norm)[0]

    模块._phonemize_cached = 只做音素化


def 转十六千赫(波形, 原采样率):
    音频 = resample_poly(波形, up = 目标采样率, down = 原采样率).astype(np.float32)
    return np.clip(音频, -1.0, 1.0)


def 主函数():
    解析器 = argparse.ArgumentParser(description = "VieNeu-TTS-v3-Turbo 合成")
    解析器.add_argument("--input", required = True, help = "合成任务 JSONL")
    解析器.add_argument("--output-dir", required = True, help = "FLAC 输出目录")
    解析器.add_argument("--batch-size", type = int, default = 16, help = "GPU 上每次前向的文本块数")
    解析器.add_argument("--results", default = None, help = "结果 JSONL，默认 <output-dir>/results.jsonl")
    解析器.add_argument("--temperature", type = float, default = 0.8, help = "采样温度，模型卡推荐 0.8")
    解析器.add_argument("--device", default = "auto", help = "auto、cuda 或 cpu")
    参数 = 解析器.parse_args()

    import torch
    from vieneu import Vieneu

    输出目录 = Path(参数.output_dir)
    输出目录.mkdir(parents = True, exist_ok = True)
    结果路径 = Path(参数.results) if 参数.results else 输出目录 / "results.jsonl"

    _关闭文本规范化()
    tts = Vieneu(
        mode = "v3turbo",
        device = 参数.device,
        max_batch_size = 参数.batch_size,
    )
    音色名集合 = {名字 for _, 名字 in tts.list_preset_voices()}

    任务 = [json.loads(行) for 行 in open(参数.input, encoding = "utf-8") if 行.strip()]
    结果表 = []
    分组 = {}  # (模式, 音色或参考音频) -> 任务序号列表
    for 序号, 条目 in enumerate(任务):
        结果 = dict(条目)
        结果.update(tts_model = "vieneu-v3-turbo", audio_path = None, duration_s = None, error = None)
        结果表.append(结果)
        模式 = 条目.get("mode", "builtin")
        文本 = 条目.get("text", "")
        音色 = 条目.get("builtin_name") or 条目.get("voice_id")
        if 条目.get("lang") != "vi":
            结果.update(status = "unsupported_lang", error = f"不支持的语言 {条目.get('lang')}")
        elif not 文本.strip():
            结果.update(status = "failed", error = "文本为空")
        elif 模式 == "builtin":
            if 音色 not in 音色名集合:
                结果.update(status = "failed", error = f"未知内置音色 {音色}")
            else:
                分组.setdefault(("builtin", 音色), []).append(序号)
        elif 模式 == "clone":
            参考 = 条目.get("ref_audio")
            if not 参考 or not Path(参考).is_file():
                结果.update(status = "failed", error = f"参考音频不存在 {参考}")
            else:
                分组.setdefault(("clone", 参考), []).append(序号)
        else:
            结果.update(status = "failed", error = f"未知 mode {模式}")

    开始 = time.perf_counter()
    音频总秒数 = 0.0
    for (模式, 键), 序号列表 in 分组.items():
        torch.manual_seed(int(任务[序号列表[0]].get("seed", 0) or 0))
        关键字 = dict(voice = 键) if 模式 == "builtin" else dict(ref_audio = 键, denoise = True)
        try:
            波形列表 = tts.infer_batch(
                texts = [任务[i]["text"] for i in 序号列表],
                batch_size = 参数.batch_size,
                temperature = 参数.temperature,
                apply_watermark = False,
                **关键字,
            )
        except Exception as 异常:
            for i in 序号列表:
                结果表[i].update(status = "failed", error = f"{type(异常).__name__}: {异常}")
            continue
        for i, 波形 in zip(序号列表, 波形列表):
            if len(波形) == 0:
                结果表[i].update(status = "failed", error = "模型没有生成音频")
                continue
            音频 = 转十六千赫(波形 = np.asarray(波形, dtype = np.float32), 原采样率 = tts.sample_rate)
            路径 = 输出目录 / f"{结果表[i]['sample_id']}.flac"
            sf.write(file = str(路径), data = 音频, samplerate = 目标采样率, subtype = "PCM_16")
            结果表[i].update(status = "ok", audio_path = str(路径), duration_s = round(len(音频) / 目标采样率, 3))
            音频总秒数 += len(音频) / 目标采样率
    用时 = time.perf_counter() - 开始

    with open(结果路径, "w", encoding = "utf-8") as 文件:
        for 结果 in 结果表:
            文件.write(json.dumps(结果, ensure_ascii = False) + "\n")

    峰值 = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    成功 = sum(1 for r in 结果表 if r["status"] == "ok")
    print(
        f"完成 {成功}/{len(任务)} 条，音频 {音频总秒数:.1f}s，用时 {用时:.1f}s，"
        f"实时倍率 {音频总秒数 / max(用时, 1e-9):.1f}x，batch_size={参数.batch_size}，显存峰值 {峰值:.2f} GiB",
        file = sys.stderr,
    )


if __name__ == "__main__":
    主函数()
