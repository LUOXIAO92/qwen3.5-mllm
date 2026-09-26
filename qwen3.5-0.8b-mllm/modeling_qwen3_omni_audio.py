from torch import nn
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
    Qwen3OmniMoePreTrainedModel,
)

from .configuration_qwen3_omni_audio import Qwen3OmniAudioTowerConfig


class Qwen3OmniAudioTowerModel(Qwen3OmniMoePreTrainedModel):
    config_class = Qwen3OmniAudioTowerConfig
    config: Qwen3OmniAudioTowerConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    input_modalities = ("audio",)
    _no_split_modules = ["Qwen3OmniMoeAudioEncoderLayer"]

    def __init__(self, config):
        super().__init__(config = config)
        self.model = nn.Module()
        self.model.audio_tower = Qwen3OmniMoeAudioEncoder(
            config = config.audio_config,
        )
        self.post_init()

    def forward(
        self,
        input_features = None,
        feature_lens = None,
        **kwargs,
    ):
        return self.model.audio_tower(
            input_features = input_features,
            feature_lens = feature_lens,
            **kwargs,
        )
