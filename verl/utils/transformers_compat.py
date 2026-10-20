# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Compatibility utilities for different versions of transformers library.
"""

from transformers import AutoModelForImageTextToText
from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask  # noqa: F401


def get_auto_model_for_vision2seq():
    """The pinned Transformers 5 model API."""
    return AutoModelForImageTextToText


def unpack_visual_output(visual_output):
    """Unpack the output from the visual encoder, handling both tuple and object return types.

    Newer versions of transformers return an object with `pooler_output` and `deepstack_features`
    attributes instead of a plain tuple.
    """
    if hasattr(visual_output, "pooler_output"):
        # For newer versions(>=5.0.0) of transformers, return the pooler_output and deepstack_features
        if hasattr(visual_output, "deepstack_features"):
            return visual_output.pooler_output, visual_output.deepstack_features
        else:
            return visual_output.pooler_output, None
    if isinstance(visual_output, tuple):
        return visual_output
    else:
        return visual_output, None


def drop_tied_target_keys(state_dict, model, model_config) -> None:
    """Drop tied alias keys (e.g. ``lm_head.weight``) from ``state_dict``.

    FSDP gather and the model merger materialise one independent CPU tensor
    per state_dict key, which defeats ``save_pretrained``'s storage-pointer-
    based dedup. When both keys end up in safetensors, ``transformers>=5``
    silently refuses to re-tie on reload. Detects aliases by ``Parameter``
    identity after ``tie_weights()``.

    Only drops the alias when the canonical name has been confirmed to exist
    in ``state_dict``; otherwise the alias is promoted to canonical so we
    never accidentally remove every entry for a tied parameter.
    """
    if not getattr(model_config, "tie_word_embeddings", False):
        return
    model.tie_weights()
    canonical_by_id: dict[int, str] = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        pid = id(param)
        if pid not in canonical_by_id:
            canonical_by_id[pid] = name
            continue
        if canonical_by_id[pid] in state_dict:
            state_dict.pop(name, None)
        else:
            # Canonical name absent: promote the alias instead of erasing it.
            canonical_by_id[pid] = name
