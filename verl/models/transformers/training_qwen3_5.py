"""Owned Qwen training classes; imported Transformers classes are never modified.

Only the multimodal embedding path, offloaded vision positional embeddings and
training output/loss differ from native Qwen. Text layers and attention remain
native. Module names match native checkpoints and PEFT adapter targets exactly.
"""

from torch import nn
from transformers import AutoModel
from transformers.models.qwen3_5 import modeling_qwen3_5 as dense
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as moe

from . import qwen3_5 as operations


class TrainingVisionModel(dense.Qwen3_5VisionModel):
    fast_pos_embed_interpolate = operations.fast_pos_embed_interpolate


class NativeMoeExperts:
    @classmethod
    def _can_set_experts_implementation(cls):
        # Transformers detects this by scanning the defining module's source.
        # These classes retain native experts, so ask their native owner directly.
        return moe.Qwen3_5MoePreTrainedModel._can_set_experts_implementation()


class TrainingMoeVisionModel(NativeMoeExperts, moe.Qwen3_5MoeVisionModel):
    fast_pos_embed_interpolate = operations.fast_pos_embed_interpolate


class TrainingModel(dense.Qwen3_5Model):
    def __init__(self, config):
        dense.Qwen3_5PreTrainedModel.__init__(self, config)
        self.visual = TrainingVisionModel._from_config(config.vision_config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()

    forward = operations.qwen3_5_base_forward


class TrainingMoeModel(NativeMoeExperts, moe.Qwen3_5MoeModel):
    def __init__(self, config):
        moe.Qwen3_5MoePreTrainedModel.__init__(self, config)
        self.visual = TrainingMoeVisionModel._from_config(config.vision_config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()

    forward = operations.qwen3_5_base_forward


class TrainingOutput:
    """Select a loss per model instance, without replacing any callable at runtime."""

    def forward(self, input_ids=None, labels=None, temperature=1.0, shift_labels=None, **kwargs):
        forwards = {
            None: operations.forward_with_normal_backend,
            "torch": operations.forward_with_torch_backend,
            "triton": operations.forward_with_triton_backend,
        }
        return forwards[self.fused_backend](
            self,
            input_ids=input_ids,
            labels=labels,
            temperature=temperature,
            shift_labels=shift_labels,
            **kwargs,
        )

    def _initialize_output(self, config, model_class, fused_backend):
        if fused_backend not in {None, "torch", "triton"}:
            raise ValueError(f"unsupported fused loss backend: {fused_backend!r}")
        self.fused_backend = fused_backend
        self.model = model_class(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()


class VerlQwen3_5ForConditionalGeneration(TrainingOutput, dense.Qwen3_5ForConditionalGeneration):
    def __init__(self, config, fused_backend=None):
        dense.Qwen3_5PreTrainedModel.__init__(self, config)
        self._initialize_output(config, TrainingModel, fused_backend)


class VerlQwen3_5MoeForConditionalGeneration(NativeMoeExperts, TrainingOutput, moe.Qwen3_5MoeForConditionalGeneration):
    def __init__(self, config, fused_backend=None):
        moe.Qwen3_5MoePreTrainedModel.__init__(self, config)
        self._initialize_output(config, TrainingMoeModel, fused_backend)
