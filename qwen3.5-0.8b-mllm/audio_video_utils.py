# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
#
# merge_audio_video_order adapts the two-pointer merge in Transformers:
# models/qwen3_omni_moe/processing_qwen3_omni_moe.py,
# Qwen3OmniMoeProcessor.replace_multimodal_special_tokens.
# It compares unrounded timestamps in seconds and returns modality/index pairs.

import numpy as np
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    _get_feat_extract_output_lengths,
)


def get_audio_token_timestamps(
    feature_length,
    n_window,
    hop_length,
    sampling_rate,
):
    """按原音频塔的分块和三层卷积，计算输出位置对应的秒数。"""
    chunk_frames = n_window * 2
    timestamps = []
    for chunk_start in range(0, int(feature_length), chunk_frames):
        chunk_length = min(chunk_frames, int(feature_length) - chunk_start)
        output_length = int(_get_feat_extract_output_lengths(
            input_lengths = chunk_length,
            n_window = n_window,
        ))
        frame_indices = chunk_start + np.arange(stop = output_length) * (2 ** 3)
        timestamps.extend((frame_indices * hop_length / sampling_rate).tolist())
    return timestamps


def merge_audio_video_order(audio_times, video_times):
    """按未舍入的秒数归并两路位置，时间相等时视频先放。"""
    video_index = 0
    audio_index = 0
    order = []
    while video_index < len(video_times) and audio_index < len(audio_times):
        if video_times[video_index] <= audio_times[audio_index]:
            order.append(("video", video_index))
            video_index += 1
        else:
            order.append(("audio", audio_index))
            audio_index += 1
    order.extend(("video", index) for index in range(video_index, len(video_times)))
    order.extend(("audio", index) for index in range(audio_index, len(audio_times)))
    return order
