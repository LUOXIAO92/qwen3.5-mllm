"""读取流水线配置（YAML 或 JSON），展开 {work_dir} 占位符并做基本检查。"""

import json
from pathlib import Path

DEFAULT_SEC_PER_CHAR = [0.03, 0.30]


def _expand(value, work_dir: str):
    if isinstance(value, str):
        return value.replace("{work_dir}", work_dir)
    if isinstance(value, list):
        return [_expand(value = item, work_dir = work_dir) for item in value]
    if isinstance(value, dict):
        return {key: _expand(value = item, work_dir = work_dir) for key, item in value.items()}
    return value


def load_config(path: Path, work_dir: Path) -> dict:
    path = Path(path)
    text = path.read_text(encoding = "utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml

        cfg = yaml.safe_load(text)
    else:
        cfg = json.loads(text)
    cfg = _expand(value = cfg, work_dir = str(Path(work_dir)))
    validate_config(cfg = cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    tts_models = cfg.get("tts_models", {})
    asr_models = cfg.get("asr_models", {})
    for lang, lang_cfg in cfg.get("langs", {}).items():
        for model, weight in lang_cfg.get("tts", {}).items():
            if model not in tts_models:
                raise ValueError(f"langs.{lang}.tts 引用了未定义的 TTS 模型 {model}")
            if weight < 0:
                raise ValueError(f"langs.{lang}.tts.{model} 的权重不能为负")
        for role in ("asr", "cross_asr"):
            name = lang_cfg.get(role)
            if name and name not in asr_models:
                raise ValueError(f"langs.{lang}.{role} 引用了未定义的 ASR 模型 {name}")
    for name, backend in {**tts_models, **asr_models}.items():
        if "module" not in backend or not ("env" in backend or "python" in backend):
            raise ValueError(f"{name} 需要 module 和 env（或 python）")


def sec_per_char(cfg: dict, lang: str) -> list:
    """语言的「秒 / 字符」合理范围 [下限, 上限]，没有单独配置时用默认值。"""
    lang_cfg = cfg.get("langs", {}).get(lang, {})
    return lang_cfg.get("sec_per_char") or cfg.get("default_sec_per_char") or DEFAULT_SEC_PER_CHAR
