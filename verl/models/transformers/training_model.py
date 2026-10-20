"""Explicit model construction for the pinned Flash training profile.

The profile uses data parallelism and Transformers' native packed attention. It
never installs the legacy model/attention patch dispatcher, even on failure.
"""

from importlib.metadata import version

from packaging.version import Version


def load_training_model(model_config, torch_dtype, sequence_parallel_size: int):
    installed = Version(version("transformers"))
    if not Version("5.12.1") <= installed < Version("5.13"):
        raise RuntimeError("explicit training models require transformers>=5.12.1,<5.13")
    if model_config.model_type != "language_model":
        raise ValueError("explicit training models require a language model")
    if sequence_parallel_size != 1:
        raise ValueError("Qwen hybrid training requires sequence_parallel_size=1")
    if model_config.use_liger or model_config.tiled_mlp.get("enabled", False):
        raise ValueError("explicit training models do not support Liger or tiled-MLP patching")

    model_type = model_config.hf_config.model_type
    from .training_qwen3_5 import VerlQwen3_5ForConditionalGeneration, VerlQwen3_5MoeForConditionalGeneration

    implementations = {
        "qwen3_5": VerlQwen3_5ForConditionalGeneration,
        "qwen3_5_moe": VerlQwen3_5MoeForConditionalGeneration,
    }
    if model_type not in implementations:
        raise ValueError(f"no explicit training implementation for {model_type!r}")
    backend = None
    if model_config.use_fused_kernels:
        backend = model_config.fused_kernel_options.get("impl_backend")
        if backend not in {"torch", "triton"}:
            raise ValueError(f"unsupported fused loss backend: {backend!r}")
    return implementations[model_type].from_pretrained(
        model_config.local_path,
        config=model_config.hf_config,
        dtype=torch_dtype,
        fused_backend=backend,
    )
