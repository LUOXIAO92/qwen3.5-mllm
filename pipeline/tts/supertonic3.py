"""Supertonic-3 内置音色 TTS 封装。

模型：Supertone/supertonic-3，约 99M 参数，ONNX Runtime 推理，原生输出 44.1 kHz，
本封装转为 16 kHz 单声道 FLAC。

支持的语言代码（取项目 18 种目标语言与模型 31 种语言的交集，共 13 种）：
    en ko ja ar de es fr id it pt ru tr vi
模型不支持 zh th ms fil fa（遇到这些语言时结果的 status 为 unsupported_lang）。

音色：只有 10 个内置音色，F1 F2 F3 F4 F5（女）、M1 M2 M3 M4 M5（男）。
不支持零样本克隆（克隆音色需要官方付费的 Voice Builder，本封装不使用），因此 mode 只能是 builtin。

环境：envs/tts-supertonic3（uv，Python 3.12，supertonic 1.3.1，onnxruntime 1.31.1 的 CPU 版）。
模型文件第一次运行时由 supertonic 包从 Hugging Face 下载，下载位置由环境变量
SUPERTONIC_CACHE_DIR 指定，也可以用 --model-dir 指定已有目录。

显存与吞吐：模型很小，在 CPU 上已经远快于实时，封装只使用 CPU，不占用 GPU，因此不需要 gpu.lock。
实测数据见 GitHub issue #12 的验收评论。

许可：代码 MIT，模型权重 OpenRAIL-M。OpenRAIL-M 允许使用输出训练模型，但附件 A 列出了使用限制，
使用者必须遵守：不得违法；不得伤害或剥削未成年人；不得制造或传播意在伤害他人的虚假信息；
不得生成可用于伤害个人的个人身份信息；传播机器生成的内容时必须明确声明其为机器生成；
不得诽谤、骚扰他人；不得未经同意冒充他人（深度伪造）；不得用于对个人权利产生不利影响的全自动决策；
不得用于歧视或针对特定群体；不得提供医疗建议或解读医疗结果；不得用于司法、执法、移民与庇护程序。
本项目只合成公开文本的朗读音频用于训练语音理解模型，不涉及上述用途。

文本处理：模型自带一段文本预处理（supertonic.core.UnicodeProcessor._preprocess_text），
其中 NFKD 分解是必需的，因为模型的字符表只收录分解后的字符，没有预先组合好的 é、가 等。
本封装把预处理收窄为：NFKD 分解、把字符表中没有的弯引号和破折号换成直引号和连字符、
合并空白、句末没有标点时补一个句号、加语言标记。
表情符号删除、`@` 展开为 at、`e.g.,` 展开、标点前空格修正、重复引号合并都已关闭，
这些情况下字符表以外的字符会让该条结果的 status 变为 failed，不会悄悄改动文本。
关闭位置见 `_收窄文本预处理`。

输入每行（JSONL）需要：sample_id、lang、text、mode（缺省按 builtin）、音色名
（builtin_name，其次 voice_id），可选 seed。其余字段原样带入结果。
输出：<output-dir>/<sample_id>.flac，以及 results.jsonl（可用 --results 改路径），
结果比输入多出 audio_path、duration_s、status（ok 或 failed、unsupported_lang 等）、error。
"""

import argparse
import json
import os
import sys
import time
from math import ceil
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

支持的语言 = ("en", "ko", "ja", "ar", "de", "es", "fr", "id", "it", "pt", "ru", "tr", "vi")
内置音色 = ("F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5")
目标采样率 = 16000
块间静音秒数 = 0.3


def _收窄文本预处理(处理器):
    """把模型自带的文本预处理收窄到必需的几步（NFKD、符号映射、空白、句号、语言标记）。"""

    def 原样(文本):
        return 文本

    # 以下几步会改动内容，全部关闭
    处理器._remove_emojis = 原样
    处理器._remove_special_chars = 原样
    处理器._expand_abbreviations = 原样
    处理器._fix_punctuation_spacing = 原样
    处理器._remove_duplicate_quotes = 原样


def _逐条噪声(引擎, 种子列表):
    """让每条文本用自己的种子生成初始噪声，结果与批内其他条目无关，可复现。"""

    def sample_noisy_latent(duration):
        批大小 = len(duration)
        采样数 = (duration * 引擎.sample_rate).astype(np.int64)
        块 = 引擎.base_chunk_size * 引擎.chunk_compress_factor
        最大长度 = int(ceil(float(duration.max()) * 引擎.sample_rate / 块))
        维度 = 引擎.ldim * 引擎.chunk_compress_factor
        噪声 = np.zeros((批大小, 维度, 最大长度), dtype=np.float32)
        掩码 = np.zeros((批大小, 1, 最大长度), dtype=np.float32)
        for i in range(批大小):
            长度 = int(ceil(int(采样数[i]) / 块))
            长度 = min(长度, 最大长度)
            生成器 = np.random.RandomState(种子列表[i])
            噪声[i, :, :长度] = 生成器.randn(维度, 长度).astype(np.float32)
            掩码[i, :, :长度] = 1.0
        return 噪声, 掩码

    引擎.sample_noisy_latent = sample_noisy_latent


def 加载引擎(model_dir, threads):
    from supertonic import TTS

    tts = TTS(
        model = "supertonic-3",
        model_dir = model_dir,
        auto_download = True,
        intra_op_num_threads = threads,
    )
    _收窄文本预处理(tts.model.text_processor)
    return tts


def 转十六千赫(波形, 原采样率):
    音频 = resample_poly(波形, up = 目标采样率, down = 原采样率).astype(np.float32)
    return np.clip(音频, -1.0, 1.0)


def 合成(tts, 单元列表, 风格表, 步数, 语速):
    """合成一批同语言的文本块。单元是 (文本, 种子, 音色名)，返回每块的 16 kHz 波形。"""
    from supertonic.core import Style

    风格 = Style(
        style_ttl_onnx = np.concatenate([风格表[u[2]].ttl for u in 单元列表], axis = 0),
        style_dp_onnx = np.concatenate([风格表[u[2]].dp for u in 单元列表], axis = 0),
    )
    _逐条噪声(引擎 = tts.model, 种子列表 = [u[1] for u in 单元列表])
    波形, 时长 = tts.model(
        text_list = [u[0] for u in 单元列表],
        style = 风格,
        total_step = 步数,
        speed = 语速,
        lang = 单元列表[0][3],
    )
    结果 = []
    for i in range(len(单元列表)):
        采样数 = int(float(时长[i]) * tts.sample_rate)
        结果.append(转十六千赫(波形 = 波形[i, :采样数], 原采样率 = tts.sample_rate))
    return 结果


def 主函数():
    解析器 = argparse.ArgumentParser(description = "Supertonic-3 内置音色合成")
    解析器.add_argument("--input", required = True, help = "合成任务 JSONL")
    解析器.add_argument("--output-dir", required = True, help = "FLAC 输出目录")
    解析器.add_argument("--batch-size", type = int, default = 16, help = "每批文本块数")
    解析器.add_argument("--results", default = None, help = "结果 JSONL，默认 <output-dir>/results.jsonl")
    解析器.add_argument("--model-dir", default = None, help = "模型目录，默认读 SUPERTONIC_CACHE_DIR")
    解析器.add_argument("--steps", type = int, default = 8, help = "扩散步数")
    解析器.add_argument("--speed", type = float, default = 1.05, help = "语速倍数")
    解析器.add_argument("--threads", type = int, default = None, help = "ONNX Runtime 线程数")
    参数 = 解析器.parse_args()

    from supertonic.config import DEFAULT_MAX_CHUNK_LENGTH, DEFAULT_MAX_CHUNK_LENGTH_KO
    from supertonic.utils import chunk_text

    输出目录 = Path(参数.output_dir)
    输出目录.mkdir(parents = True, exist_ok = True)
    结果路径 = Path(参数.results) if 参数.results else 输出目录 / "results.jsonl"

    任务 = [json.loads(行) for 行 in open(参数.input, encoding = "utf-8") if 行.strip()]
    模型目录 = 参数.model_dir or os.environ.get("SUPERTONIC_CACHE_DIR")
    tts = 加载引擎(model_dir = 模型目录, threads = 参数.threads)
    风格表 = {名字: tts.get_voice_style(voice_name = 名字) for 名字 in 内置音色}

    结果表 = {}
    单元 = []  # (任务序号, 块序号, 文本, 种子, 音色, 语言)
    块数 = {}
    for 序号, 条目 in enumerate(任务):
        结果 = dict(条目)
        结果.update(tts_model = "supertonic-3", audio_path = None, duration_s = None, error = None)
        结果表[序号] = 结果
        音色 = 条目.get("builtin_name") or 条目.get("voice_id")
        模式 = 条目.get("mode", "builtin")
        语言 = 条目.get("lang")
        文本 = 条目.get("text", "")
        if 模式 != "builtin":
            结果.update(status = "failed", error = f"supertonic-3 只支持 builtin 模式，收到 {模式}")
        elif 语言 not in 支持的语言:
            结果.update(status = "unsupported_lang", error = f"不支持的语言 {语言}")
        elif 音色 not in 风格表:
            结果.update(status = "failed", error = f"未知内置音色 {音色}")
        elif not 文本.strip():
            结果.update(status = "failed", error = "文本为空")
        else:
            合法, 不支持字符 = tts.model.text_processor.validate_text(文本)
            if not 合法:
                结果.update(status = "failed", error = f"字符表之外的字符 {不支持字符}")
                continue
            块长 = DEFAULT_MAX_CHUNK_LENGTH_KO if 语言 == "ko" else DEFAULT_MAX_CHUNK_LENGTH
            各块 = chunk_text(文本, 块长)
            块数[序号] = len(各块)
            种子 = int(条目.get("seed", 0) or 0)
            for 块序号, 块 in enumerate(各块):
                单元.append((序号, 块序号, 块, 种子 + 块序号, 音色, 语言))

    # 同语言、相近长度的块放进同一批，减少填充
    单元.sort(key = lambda u: (u[5], len(u[2])))
    各块波形 = {}
    开始 = time.perf_counter()
    音频总秒数 = 0.0
    i = 0
    while i < len(单元):
        语言 = 单元[i][5]
        批 = []
        while i < len(单元) and 单元[i][5] == 语言 and len(批) < 参数.batch_size:
            批.append(单元[i])
            i += 1
        波形列表 = 合成(
            tts = tts,
            单元列表 = [(u[2], u[3], u[4], u[5]) for u in 批],
            风格表 = 风格表,
            步数 = 参数.steps,
            语速 = 参数.speed,
        )
        for u, 波形 in zip(批, 波形列表):
            各块波形[(u[0], u[1])] = 波形
            音频总秒数 += len(波形) / 目标采样率
    用时 = time.perf_counter() - 开始

    静音 = np.zeros(int(块间静音秒数 * 目标采样率), dtype = np.float32)
    for 序号, 数量 in 块数.items():
        片段 = []
        for k in range(数量):
            if k:
                片段.append(静音)
            片段.append(各块波形[(序号, k)])
        音频 = np.concatenate(片段)
        结果 = 结果表[序号]
        路径 = 输出目录 / f"{结果['sample_id']}.flac"
        sf.write(file = str(路径), data = 音频, samplerate = 目标采样率, subtype = "PCM_16")
        结果.update(status = "ok", audio_path = str(路径), duration_s = round(len(音频) / 目标采样率, 3))

    with open(结果路径, "w", encoding = "utf-8") as 文件:
        for 序号 in range(len(任务)):
            文件.write(json.dumps(结果表[序号], ensure_ascii = False) + "\n")

    倍率 = 音频总秒数 / 用时 if 用时 > 0 else 0.0
    print(
        f"完成 {len(块数)}/{len(任务)} 条，音频 {音频总秒数:.1f}s，用时 {用时:.1f}s，"
        f"实时倍率 {倍率:.1f}x，batch_size={参数.batch_size}",
        file = sys.stderr,
    )


if __name__ == "__main__":
    主函数()
