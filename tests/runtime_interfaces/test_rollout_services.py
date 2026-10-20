"""Owned rollout services with real tensors and isolated dependency boundaries."""

import ast
import importlib.util
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stub(monkeypatch, name, **attributes):
    module = types.ModuleType(name)
    vars(module).update(attributes)
    monkeypatch.setitem(sys.modules, name, module)


def test_vocabulary_hooks_are_per_model_after_lora_and_removable(monkeypatch):
    class Processor(torch.nn.Module):
        def forward(self, logits):
            return logits

    class LoraProcessor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = Processor()

        def forward(self, logits):
            return self.base_layer(logits) + 3 if logits is not None else None

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits_processor = LoraProcessor()

        def compute_logits(self, logits):
            return self.logits_processor(logits)

    stub(monkeypatch, "vllm.lora.layers.logits_processor", LogitsProcessorWithLoRA=LoraProcessor)
    stub(monkeypatch, "vllm.model_executor.layers.logits_processor", LogitsProcessor=Processor)
    service = load_file("vocabulary", ROOT / "verl/utils/vllm/vocabulary.py")
    first, second = Model(), Model()
    original = Model.compute_logits
    handles = service.configure_model_vocabulary(first, 3)
    assert len(handles) == 1
    output = first.compute_logits(torch.arange(6.0).reshape(1, -1))
    torch.testing.assert_close(output[:, :3], torch.tensor([[3.0, 4.0, 5.0]]))
    assert torch.isneginf(output[:, 3:]).all()
    expected_logprobs = torch.tensor([[3.0, 4.0, 5.0]]).log_softmax(-1)
    torch.testing.assert_close(output.log_softmax(-1)[:, :3], expected_logprobs)
    assert second.compute_logits(torch.zeros(1, 6)).isfinite().all()
    assert first.compute_logits(None) is None
    assert Model.compute_logits is original
    handles[0].remove()
    assert first.compute_logits(torch.zeros(1, 6)).isfinite().all()
    with pytest.raises(ValueError, match="positive integer"):
        service.configure_model_vocabulary(first, 0)
    with pytest.raises(RuntimeError, match="no supported logits processor"):
        service.configure_model_vocabulary(torch.nn.Linear(2, 2), 3)


@pytest.fixture
def adapter_runtime(monkeypatch):
    import msgspec

    class Request(msgspec.Struct):
        lora_int_id: int
        load_inplace: bool = False

    class Cache:
        def __init__(self):
            self.model = SimpleNamespace(hf_to_vllm_mapper=object(), lora_skip_prefixes=["mtp."])
            self.values = {}
            self.capacity = 1
            self.active = None

        def __len__(self):
            return len(self.values)

        def list_adapters(self):
            return self.values

        def remove_adapter(self, key):
            self.values.pop(key, None)

        def remove_oldest_adapter(self):
            self.values.pop(next(iter(self.values)))

        def add_adapter(self, value):
            self.values[value.id] = value
            return True

        def get_adapter(self, key):
            return self.values.get(key)

        def activate_adapter(self, key):
            assert key in self.values
            self.active = key

    class NativeManager:
        def __init__(self, config, device, embedding_modules, lora_model_cls):
            self.device, self.embedding_modules = device, embedding_modules
            self._lora_model_cls = lora_model_cls
            self.lora_config = config.lora_config
            self.vocab_size = config.model_config.get_vocab_size()

        def _load_adapter(self, request):
            return "native-file-loader"

    # Optional qualification executes the exact pinned vLLM cache lifecycle, with
    # only CUDA/cache storage dependencies substituted. No implementation copy.
    source = os.getenv("VERL_TEST_VLLM_WORKER_MANAGER")
    if source:
        tree = ast.parse(Path(source).read_text())
        classes = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name in {"WorkerLoRAManager", "LRUCacheWorkerLoRAManager"}
        ]
        namespace = dict(
            contextmanager=contextmanager,
            LoRAModelManager=Cache,
            LRUCacheLoRAModelManager=Cache,
            LoRAModel=object,
            torch=torch,
        )
        program = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *classes],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(program), source, "exec"), namespace)
        NativeManager = namespace["LRUCacheWorkerLoRAManager"]

    seen = []

    class Helper:
        @classmethod
        def from_dict(cls, value):
            return cls()

        def validate_legal(self, config):
            if config.invalid:
                raise ValueError("invalid adapter")

    class Model:
        @staticmethod
        def from_lora_tensors(**kwargs):
            seen.append(kwargs)
            return SimpleNamespace(id=kwargs["lora_model_id"], tensors=kwargs["tensors"])

    stub(monkeypatch, "vllm.lora.peft_helper", PEFTHelper=Helper)
    stub(monkeypatch, "vllm.lora.request", LoRARequest=Request)
    stub(monkeypatch, "vllm.lora.worker_manager", LRUCacheWorkerLoRAManager=NativeManager)
    stub(monkeypatch, "verl.third_party.vllm", get_version=lambda _: "0.19.1")
    service = load_file("tensor_adapter_service", ROOT / "verl/utils/vllm/utils.py")
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=32),
        lora_config=SimpleNamespace(lora_dtype=torch.float32, invalid=False),
        model_config=SimpleNamespace(
            get_vocab_size=lambda: 64,
            is_encoder_decoder=False,
            hf_config=SimpleNamespace(get_text_config=lambda: SimpleNamespace(max_position_embeddings=32)),
        ),
    )
    native = NativeManager(config, "cpu", {}, Model)
    native._adapter_manager = Cache()
    runner = SimpleNamespace(vllm_config=config, lora_manager=native)
    return service, runner, seen, NativeManager


def test_tensor_adapter_loader_preserves_native_methods_and_uses_owned_cache(adapter_runtime):
    service, runner, seen, native_type = adapter_runtime
    original = native_type._load_adapter
    owned = service.TensorLoRAWorkerManager.for_worker(runner)
    assert owned is not runner.lora_manager
    assert owned._adapter_manager is runner.lora_manager._adapter_manager
    tensors = {"model.layers.0.q_proj.lora_A.weight": torch.randn(2, 4)}
    request = service.TensorLoRARequest(123, load_inplace=True, peft_config={"r": 2}, lora_tensors=tensors)
    result = owned._load_adapter(request)
    assert result.tensors is tensors
    assert seen[0]["weights_mapper"] is runner.lora_manager._adapter_manager.model.hf_to_vllm_mapper
    assert seen[0]["skip_prefixes"] == ["mtp."]
    assert seen[0]["model_vocab_size"] == 64
    assert native_type._load_adapter is original
    runner.vllm_config.lora_config.invalid = True
    with pytest.raises(ValueError, match="invalid adapter"):
        owned._load_adapter(request)
    assert len(seen) == 1


@pytest.mark.skipif(not os.getenv("VERL_TEST_VLLM_WORKER_MANAGER"), reason="requires pinned vLLM source or runtime")
def test_native_vllm_cache_refreshes_same_id_and_keeps_old_adapter_on_validation_failure(adapter_runtime):
    service, runner, _, _ = adapter_runtime
    owned = service.TensorLoRAWorkerManager.for_worker(runner)
    cache = runner.lora_manager._adapter_manager
    for value in (1.0, 2.0):
        request = service.TensorLoRARequest(123, load_inplace=True, peft_config={}, lora_tensors={"a": value})
        assert owned.add_adapter(request)
        assert cache.values[123].tensors == {"a": value}
        assert cache.active == 123
        assert runner.lora_manager.list_adapters() == {123}
    runner.vllm_config.lora_config.invalid = True
    with pytest.raises(ValueError, match="invalid adapter"):
        owned.add_adapter(request)
    assert cache.values[123].tensors == {"a": 2.0}
