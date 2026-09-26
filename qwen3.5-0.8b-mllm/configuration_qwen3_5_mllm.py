from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
)


class Qwen3_5MLLMConfig(Qwen3_5Config):
    """在 Qwen3.5 配置中加入音频塔配置。"""

    model_type = "qwen3_5_mllm"
    sub_configs = {
        **Qwen3_5Config.sub_configs,
        "audio_config": Qwen3OmniMoeAudioEncoderConfig,
    }

    def __init__(
        self,
        audio_config = None,
        audio_token_id = None,
        audio_start_token_id = None,
        audio_end_token_id = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if audio_config is None:
            audio_config = Qwen3OmniMoeAudioEncoderConfig()
        elif isinstance(audio_config, dict):
            audio_config = Qwen3OmniMoeAudioEncoderConfig.from_dict(
                config_dict = audio_config,
            )
        self.audio_config = audio_config
        self.audio_token_id = audio_token_id
        self.audio_start_token_id = audio_start_token_id
        self.audio_end_token_id = audio_end_token_id
