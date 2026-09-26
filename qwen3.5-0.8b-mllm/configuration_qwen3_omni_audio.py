from transformers import PretrainedConfig
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
)


class Qwen3OmniAudioTowerConfig(PretrainedConfig):
    model_type = "qwen3_omni_audio_tower"
    sub_configs = {"audio_config": Qwen3OmniMoeAudioEncoderConfig}

    def __init__(self, audio_config = None, **kwargs):
        super().__init__(**kwargs)
        if audio_config is None:
            audio_config = Qwen3OmniMoeAudioEncoderConfig()
        elif isinstance(audio_config, dict):
            audio_config = Qwen3OmniMoeAudioEncoderConfig.from_dict(
                config_dict = audio_config,
            )
        self.audio_config = audio_config
