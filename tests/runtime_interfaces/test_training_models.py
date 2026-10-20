"""Real tiny Transformers models: isolation, numerical parity and checkpoint ABI."""

import copy
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def runtime(monkeypatch):
    # Only driver/device package initialization is isolated. Model construction,
    # forwards, fused torch loss and autograd execute the actual implementations.
    original = {key: value for key, value in sys.modules.items() if key == "verl" or key.startswith("verl.")}
    for name in ("verl", "verl.utils"):
        package = types.ModuleType(name)
        package.__path__ = [str(ROOT.joinpath(*name.split(".")))]
        monkeypatch.setitem(sys.modules, name, package)
    module = importlib.import_module("verl.models.transformers.training_qwen3_5")
    yield module
    for key in list(sys.modules):
        if (key == "verl" or key.startswith("verl.")) and key not in original:
            sys.modules.pop(key, None)
    sys.modules.update(original)


def tiny_config(moe=False, hybrid=False):
    from transformers import Qwen3_5Config, Qwen3_5MoeConfig

    config_class = Qwen3_5MoeConfig if moe else Qwen3_5Config
    config = config_class(
        text_config=dict(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            layer_types=["linear_attention" if hybrid else "full_attention", "full_attention"],
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000,
                "partial_rotary_factor": 1.0,
                "mrope_section": [1, 1, 2],
            },
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
            num_experts=4,
            num_experts_per_tok=2,
        ),
        vision_config=dict(
            depth=1,
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            out_hidden_size=32,
            patch_size=2,
            temporal_patch_size=1,
            spatial_merge_size=2,
            num_position_embeddings=16,
        ),
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
    )
    config._attn_implementation = "eager"
    config.text_config.use_cache = False
    config.architectures = ["Qwen3_5MoeForConditionalGeneration" if moe else "Qwen3_5ForConditionalGeneration"]
    return config


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("hybrid", [False, True])
def test_models_preserve_legacy_outputs_gradients_and_native_classes(runtime, monkeypatch, moe, multimodal, hybrid):
    native = runtime.moe if moe else runtime.dense
    prefix = "Qwen3_5Moe" if moe else "Qwen3_5"
    root_class = getattr(native, prefix + "ForConditionalGeneration")
    model_class = getattr(native, prefix + "Model")
    vision_class = getattr(native, prefix + "VisionModel")
    owned_class = getattr(runtime, "Verl" + prefix + "ForConditionalGeneration")
    originals = (root_class.forward, model_class.forward, vision_class.fast_pos_embed_interpolate)
    from transformers.integrations import flash_attention

    attention = flash_attention._flash_attention_forward
    config = tiny_config(moe, hybrid)
    torch.manual_seed(41)
    baseline = root_class(copy.deepcopy(config))
    owned = owned_class(copy.deepcopy(config), fused_backend="torch")
    assert set(owned.state_dict()) == set(baseline.state_dict())
    owned.load_state_dict(baseline.state_dict(), strict=True)
    inputs = dict(input_ids=torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]), use_cache=False)
    if multimodal:
        inputs.update(
            input_ids=torch.tensor([[1, 60, 60, 60, 60, 2, 3, 4]]),
            pixel_values=torch.randn(16, 12),
            image_grid_thw=torch.tensor([[1, 4, 4]]),
        )
    inputs["shift_labels"] = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
    with monkeypatch.context() as legacy:
        legacy.setattr(root_class, "forward", runtime.operations.forward_with_torch_backend)
        legacy.setattr(model_class, "forward", runtime.operations.qwen3_5_base_forward)
        legacy.setattr(vision_class, "fast_pos_embed_interpolate", runtime.operations.fast_pos_embed_interpolate)
        expected = baseline(**inputs)
        (-expected.log_probs.mean() + expected.entropy.mean() * 0.01).backward()
    actual = owned(**inputs)
    (-actual.log_probs.mean() + actual.entropy.mean() * 0.01).backward()
    torch.testing.assert_close(actual.log_probs, expected.log_probs)
    torch.testing.assert_close(actual.entropy, expected.entropy)
    for (name, param), (other_name, other) in zip(owned.named_parameters(), baseline.named_parameters(), strict=True):
        assert name == other_name
        assert (param.grad is None) == (other.grad is None), name
        if param.grad is not None:
            torch.testing.assert_close(param.grad, other.grad, msg=name)
    assert originals == (root_class.forward, model_class.forward, vision_class.fast_pos_embed_interpolate)
    assert flash_attention._flash_attention_forward is attention
    assert "verl.models.transformers.monkey_patch" not in sys.modules


@pytest.mark.parametrize("moe", [False, True])
def test_loads_native_checkpoint_with_unchanged_names_and_per_instance_backend(runtime, tmp_path, moe):
    factory = importlib.import_module("verl.models.transformers.training_model")
    native = runtime.moe if moe else runtime.dense
    root = native.Qwen3_5MoeForConditionalGeneration if moe else native.Qwen3_5ForConditionalGeneration
    model = root(tiny_config(moe))
    model.save_pretrained(tmp_path)
    options = SimpleNamespace(
        model_type="language_model",
        hf_config=copy.deepcopy(model.config),
        local_path=str(tmp_path),
        use_liger=False,
        tiled_mlp={},
        use_fused_kernels=True,
        fused_kernel_options={"impl_backend": "torch"},
    )
    first = factory.load_training_model(options, torch.float32, 1)
    options.fused_kernel_options = {"impl_backend": "triton"}
    second = factory.load_training_model(options, torch.float32, 1)
    assert first.fused_backend == "torch"
    assert second.fused_backend == "triton"
    assert first.config.architectures == model.config.architectures
    for name, value in model.state_dict().items():
        torch.testing.assert_close(first.state_dict()[name], value)
    options.fused_kernel_options = {"impl_backend": "unsupported"}
    with pytest.raises(ValueError, match="unsupported fused loss"):
        factory.load_training_model(options, torch.float32, 1)
    with pytest.raises(ValueError, match="sequence_parallel_size=1"):
        factory.load_training_model(options, torch.float32, 2)


@pytest.mark.parametrize("moe", [False, True])
def test_peft_adapter_trains_and_reloads_without_changing_checkpoint_targets(runtime, tmp_path, moe):
    from abc import ABC

    from peft import LoraConfig, PeftModel, get_peft_model

    cls = runtime.VerlQwen3_5MoeForConditionalGeneration if moe else runtime.VerlQwen3_5ForConditionalGeneration
    base = cls(tiny_config(moe, hybrid=True), fused_backend="torch")
    original = copy.deepcopy(base.state_dict())
    trained = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"], task_type="CAUSAL_LM"))
    # This normal training model does not enter Verl's conditional FSDP/ABC replacement.
    assert not isinstance(trained, ABC)
    inputs = dict(input_ids=torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]), use_cache=False)
    optimizer = torch.optim.SGD([p for p in trained.parameters() if p.requires_grad], lr=0.1)
    (-trained(**inputs).log_probs.mean()).backward()
    adapters = [p for name, p in trained.named_parameters() if "lora_B" in name]
    assert adapters and all(p.grad is not None and p.grad.abs().sum() > 0 for p in adapters)
    optimizer.step()
    trained.eval()
    trained.save_pretrained(tmp_path)
    fresh = cls(tiny_config(moe, hybrid=True), fused_backend="torch")
    fresh.load_state_dict(original, strict=True)
    restored = PeftModel.from_pretrained(fresh, tmp_path)
    with torch.no_grad():
        torch.testing.assert_close(trained(**inputs).log_probs, restored(**inputs).log_probs)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"model_type": "value_model"}, "language model"),
        ({"use_liger": True}, "Liger"),
        ({"tiled_mlp": {"enabled": True}}, "tiled-MLP"),
        ({"hf_config": SimpleNamespace(model_type="unqualified")}, "no explicit training implementation"),
    ],
)
def test_unsupported_profiles_fail_before_loading_weights(runtime, change, message):
    factory = importlib.import_module("verl.models.transformers.training_model")
    config = dict(
        model_type="language_model",
        hf_config=tiny_config(),
        local_path="/does-not-exist",
        use_liger=False,
        tiled_mlp={},
        use_fused_kernels=True,
        fused_kernel_options={"impl_backend": "torch"},
    )
    config.update(change)
    with pytest.raises(ValueError, match=message):
        factory.load_training_model(SimpleNamespace(**config), torch.float32, 1)


def test_unqualified_transformers_version_fails_before_model_loading(runtime, monkeypatch):
    factory = importlib.import_module("verl.models.transformers.training_model")
    monkeypatch.setattr(factory, "version", lambda _: "5.10.2")
    with pytest.raises(RuntimeError, match="require transformers"):
        factory.load_training_model(None, torch.float32, 1)
