"""CPU contract tests; no Ray cluster, model download, or CUDA required."""

import ast
import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("custom", [None, "custom.Engine", "invalid.Engine", "invalid.value"])
def test_engine_selection_is_per_recipe_and_rejects_non_engines(monkeypatch, custom):
    class BaseEngine:
        pass

    class CustomEngine(BaseEngine):
        pass

    original = type("Original", (BaseEngine,), {})
    registry = SimpleNamespace(get_engine_cls=lambda model, backend: original)
    values = {"custom.Engine": CustomEngine, "invalid.Engine": object, "invalid.value": 12}
    for name, attrs in {
        "verl.workers.engine": {"BaseEngine": BaseEngine, "EngineRegistry": registry},
        "verl.utils.import_utils": {"load_class_from_fqn": lambda name, _: values[name]},
    }.items():
        module = types.ModuleType(name)
        vars(module).update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    interface = load_file("runtime_interfaces", "verl/runtime_interfaces.py")
    if custom and custom.startswith("invalid"):
        with pytest.raises(TypeError, match="must inherit BaseEngine"):
            interface.resolve_engine_class({"custom_engine_class": custom}, "language_model", "fsdp2")
    else:
        selected = interface.resolve_engine_class({"custom_engine_class": custom}, "language_model", "fsdp2")
        assert selected is (CustomEngine if custom else original)
    assert interface.resolve_engine_class({}, "language_model", "fsdp2") is original


def native_generate():
    # Execute the real server method with only its vLLM boundary substituted. Do not copy
    # its implementation into a fake server: that would test the fake's guard instead.
    path = ROOT / "verl/workers/rollout/vllm_rollout/vllm_async_server.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "vLLMHttpServer")
    method = next(node for node in cls.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "generate")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {
        "normalize_token_ids": list,
        "qwen2_5_vl_dedup_image_tokens": lambda ids, _: ids,
        "SamplingParams": lambda **kw: SimpleNamespace(**kw),
        "TokensPrompt": lambda **kw: kw,
        "LoRARequest": lambda **kw: SimpleNamespace(**kw),
        "VLLM_LORA_NAME": "actor",
        "VLLM_LORA_INT_ID": 123,
        "VLLM_LORA_PATH": "adapter",
        "TokenOutput": lambda **kw: kw,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["generate"]


@pytest.mark.parametrize(("adapter", "loaded", "allowed"), [(True, [], False), (True, [123], True), (False, [], True)])
def test_native_lora_guard_checks_the_actual_request_before_generation(adapter, loaded, allowed):
    requests = []

    class Engine:
        async def list_loras(self):
            return loaded

        async def generate(self, **kwargs):
            requests.append(kwargs)
            yield SimpleNamespace(outputs=[])

    config = SimpleNamespace(max_model_len=16, response_length=4, prompt_length=4, get=lambda _, default: default)
    server = SimpleNamespace(
        config=config, model_config=SimpleNamespace(processor=None), engine=Engine(), lora_as_adapter=adapter
    )
    call = native_generate()(server, [1, 2], {}, "request")
    if allowed:
        assert asyncio.run(call)["stop_reason"] == "aborted"
        assert len(requests) == 1
        request = requests[0]["lora_request"]
        assert (request.lora_int_id if request else None) == (123 if adapter else None)
    else:
        with pytest.raises(RuntimeError, match="trained LoRA adapter is not loaded"):
            asyncio.run(call)
        assert requests == []
