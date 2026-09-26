import torch
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
    Qwen3_5PreTrainedModel,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
    SinusoidsPositionEmbedding,
    _get_feat_extract_output_lengths,
)

from .configuration_qwen3_5_mllm import Qwen3_5MLLMConfig


class Qwen3_5MLLMAudioProjector(nn.Module):
    """将音频塔输出映射到语言模型的特征维度。"""

    def __init__(self, audio_output_dim, text_hidden_size):
        super().__init__()
        self.linear_1 = nn.Linear(
            in_features = audio_output_dim,
            out_features = text_hidden_size,
            bias = True,
        )
        self.silu = nn.SiLU()
        self.linear_2 = nn.Linear(
            in_features = text_hidden_size,
            out_features = text_hidden_size,
            bias = True,
        )

    def forward(self, audio_features):
        hidden_states = self.linear_1(input = audio_features)
        hidden_states = self.silu(input = hidden_states)
        return self.linear_2(input = hidden_states)


class Qwen3_5MLLMModel(Qwen3_5Model):
    """在 Qwen3.5 的视觉和语言主干中加入音频塔与 projector。"""

    config_class = Qwen3_5MLLMConfig
    base_model_prefix = "model"
    input_modalities = ("text", "image", "video", "audio")
    _can_compile_fullgraph = False
    _no_split_modules = [
        "Qwen3_5DecoderLayer",
        "Qwen3_5VisionBlock",
        "Qwen3OmniMoeAudioEncoderLayer",
    ]

    def __init__(self, config):
        super().__init__(config = config)
        self.audio_tower = Qwen3OmniMoeAudioEncoder(config = config.audio_config)
        self.audio_projector = Qwen3_5MLLMAudioProjector(
            audio_output_dim = config.audio_config.output_dim,
            text_hidden_size = config.text_config.hidden_size,
        )
        self.post_init()

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module = module)
        if isinstance(module, SinusoidsPositionEmbedding):
            positions = module.compute_default_singular_positional_embedding()
            module.positional_embedding.copy_(other = positions)

    def get_rope_index(
        self,
        input_ids,
        mm_token_type_ids = None,
        image_grid_thw = None,
        video_grid_thw = None,
        attention_mask = None,
        **kwargs,
    ):
        if mm_token_type_ids is None:
            mm_token_type_ids = torch.zeros_like(input = input_ids)
            mm_token_type_ids[input_ids == self.config.image_token_id] = 1
            mm_token_type_ids[input_ids == self.config.video_token_id] = 2
            if self.config.audio_token_id is not None:
                mm_token_type_ids[input_ids == self.config.audio_token_id] = 3
        # 音频沿用原生的一维位置规则；保留原模态编号供特征填入使用。
        position_types = mm_token_type_ids.clone()
        position_types[position_types == 3] = 0
        return super().get_rope_index(
            input_ids = input_ids,
            mm_token_type_ids = position_types,
            image_grid_thw = image_grid_thw,
            video_grid_thw = video_grid_thw,
            attention_mask = attention_mask,
        )

    def get_audio_features(self, input_features, feature_attention_mask):
        feature_mask = feature_attention_mask.to(
            device = input_features.device,
            dtype = torch.bool,
        )
        feature_lens = feature_mask.sum(dim = -1)
        mel_features = input_features.transpose(dim0 = 1, dim1 = 2)
        packed_features = mel_features[feature_mask]
        packed_features = packed_features.transpose(dim0 = 0, dim1 = 1).contiguous()
        packed_features = packed_features.to(
            device = self.audio_tower.device,
            dtype = self.audio_tower.dtype,
        )
        audio_outputs = self.audio_tower(
            input_features = packed_features,
            feature_lens = feature_lens.to(device = self.audio_tower.device),
        )
        projector_weight = self.audio_projector.linear_1.weight
        audio_features = audio_outputs.last_hidden_state.to(
            device = projector_weight.device,
            dtype = projector_weight.dtype,
        )
        return self.audio_projector(audio_features = audio_features)

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        pixel_values = None,
        pixel_values_videos = None,
        image_grid_thw = None,
        video_grid_thw = None,
        mm_token_type_ids = None,
        input_features = None,
        feature_attention_mask = None,
        **kwargs,
    ):
        if input_features is not None:
            if input_ids is None or inputs_embeds is not None:
                raise ValueError("传入音频时，请用 input_ids 指明音频标记的位置。")
            if feature_attention_mask is None or self.config.audio_token_id is None:
                raise ValueError("音频输入需要 feature_attention_mask 和 audio_token_id。")
            audio_mask = input_ids == self.config.audio_token_id
            feature_lens = feature_attention_mask.sum(dim = -1)
            audio_lengths = _get_feat_extract_output_lengths(
                input_lengths = feature_lens,
                n_window = self.config.audio_config.n_window,
            )
            actual_lengths = audio_mask.sum(dim = -1)
            if actual_lengths.shape != audio_lengths.shape or not torch.equal(
                input = actual_lengths,
                other = audio_lengths.to(device = actual_lengths.device),
            ):
                raise ValueError("每条输入的音频标记数量必须与音频特征数量一致。")
            inputs_embeds = self.get_input_embeddings()(input = input_ids)
            audio_features = self.get_audio_features(
                input_features = input_features,
                feature_attention_mask = feature_attention_mask,
            )
            audio_features = audio_features.to(
                device = inputs_embeds.device,
                dtype = inputs_embeds.dtype,
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                mask = audio_mask.unsqueeze(dim = -1),
                source = audio_features,
            )
            if position_ids is None:
                position_ids, self.rope_deltas = self.get_rope_index(
                    input_ids = input_ids,
                    mm_token_type_ids = mm_token_type_ids,
                    image_grid_thw = image_grid_thw,
                    video_grid_thw = video_grid_thw,
                    attention_mask = attention_mask,
                )
            input_ids = None

        return super().forward(
            input_ids = input_ids,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            pixel_values = pixel_values,
            pixel_values_videos = pixel_values_videos,
            image_grid_thw = image_grid_thw,
            video_grid_thw = video_grid_thw,
            mm_token_type_ids = mm_token_type_ids,
            **kwargs,
        )


class Qwen3_5MLLMForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """使用融合后的多模态主干，沿用 Qwen3.5 的文本输出与生成方法。"""

    config_class = Qwen3_5MLLMConfig
    base_model_prefix = "model"
    input_modalities = ("text", "image", "video", "audio")
    _can_compile_fullgraph = False
    _no_split_modules = Qwen3_5MLLMModel._no_split_modules

    def __init__(self, config):
        Qwen3_5PreTrainedModel.__init__(self = self, config = config)
        self.model = Qwen3_5MLLMModel(config = config)
        self.lm_head = nn.Linear(
            in_features = config.text_config.hidden_size,
            out_features = config.text_config.vocab_size,
            bias = False,
        )
        self.post_init()

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module = module)
        if isinstance(module, SinusoidsPositionEmbedding):
            positions = module.compute_default_singular_positional_embedding()
            module.positional_embedding.copy_(other = positions)

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        labels = None,
        pixel_values = None,
        pixel_values_videos = None,
        image_grid_thw = None,
        video_grid_thw = None,
        mm_token_type_ids = None,
        input_features = None,
        feature_attention_mask = None,
        logits_to_keep = 0,
        **kwargs,
    ):
        return super().forward(
            input_ids = input_ids,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            labels = labels,
            pixel_values = pixel_values,
            pixel_values_videos = pixel_values_videos,
            image_grid_thw = image_grid_thw,
            video_grid_thw = video_grid_thw,
            mm_token_type_ids = mm_token_type_ids,
            input_features = input_features,
            feature_attention_mask = feature_attention_mask,
            logits_to_keep = logits_to_keep,
            **kwargs,
        )
