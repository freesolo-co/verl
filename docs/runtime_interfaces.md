# Explicit trainer interfaces (fork API version 2)

This fork exposes opt-in construction points for packaged training recipes. `verl.runtime_interfaces.API_VERSION` identifies this contract. Recipes select owned implementations instead of replacing module globals or dependency methods. Version 2 adds model construction and replaces the global tensor-LoRA loader override. The fork now declares Transformers >=5.12.1,<5.13 and vLLM 0.19.1 for its vLLM extra.

- `HFModelConfig.custom_engine_class` is an importable `BaseEngine` subclass. `runtime_options` is a recipe-owned serializable dictionary. `TrainingWorker` resolves the class for each worker; it does not modify `EngineRegistry`. The recipe must select an engine compatible with its configured model and backend.
- `FSDPEngine._load_model(dtype)` constructs a model within the engine's existing weight-initialization context. The default remains the legacy recipe; Flash overrides it with `load_training_model`, which constructs owned Qwen3.5 dense/MoE classes and never invokes `apply_monkey_patch`. The factory rejects unsupported architectures, sequence parallelism, Liger, tiled MLP, and invalid fused-loss backends before loading weights.
- `TaskRunner.trainer_class` selects the GRPO/PPO trainer. `SFTTrainer.tracking_class`, `SFTTrainer.checkpoint_handler_class`, and both PPO trainers' `tracking_class` select services with the original constructor signatures. The recipe owns their lifetime.
- `RayPPOTrainer._compute_advantage` and `_apply_rollout_correction` delegate to the original functions by default. Overrides must preserve batch structure and return values.
- `PPOTrainer._create_replay_buffer` constructs a buffer owned by that trainer. The caller remains responsible for closing it.
- `AgentLoopWorker._create_teacher_manager` selects a teacher service. `generate_sequences(..., trajectory_info=...)` accepts a per-row identity sidecar; its length must match the batch. Omitting it retains the existing identity generation.
- `AgentLoopWorkerTQBase` is an ordinary subclassable worker. `AgentLoopWorkerTQ` remains its Ray actor wrapper. `AgentLoopManagerTQ.worker_class` selects that wrapper per manager type.

LoRA rollout now fails before generation when the configured adapter has not been loaded. Previously that condition silently generated from the base model. Full-model rollouts remain unchanged.

The explicit Qwen classes preserve native checkpoint/module names and use the existing multimodal, vision-position and fused-loss operations as ordinary subclass methods. Text/attention modules remain native Transformers instances. Each model selects its own torch/triton loss backend. No imported class or attention registry is modified. Transformers' MoE capability probe scans the defining source module; the owned MoE subclasses explicitly delegate that probe to the native MoE class, whose experts they retain.

Rollout vocabulary filtering uses owned PyTorch forward-hook handles on logits processors, after any LoRA output and before sampling/logprob normalization. It leaves `compute_logits` unchanged and supports non-output TP ranks returning `None`. `TensorLoRAWorkerManager` is an explicitly constructed subclass sharing the existing worker's adapter cache. It overrides tensor decoding only; vLLM's native cache insertion, eviction, activation, and in-place replacement remain authoritative. Native file-backed requests continue through the native manager. The only private cache binding is isolated in `for_worker`; this is qualified against vLLM 0.19.1. Failed adapter validation preserves the previous cache entry.

Old Transformers 4.52–4.57 branches, the Qwen3-VL 4.57.3 router workaround, old auto-model import fallbacks and old tensor-LoRA API fallbacks have been deleted. Default legacy recipes for other architectures still have a separate patch dispatcher. NPU/Megatron and optional quantization integrations remain conditional; removing their support is outside the Flash profile.

The MoE weight-loader repair is deliberately retained. In vLLM 0.19.1, post-load quantization/kernel preparation can replace expert Parameters, losing the attached loader. Its 0.8.2 origin comment is not evidence of obsolescence. Removing it needs GPU coverage of initial/base weight synchronization and reload for every selected MoE kernel. PyTorch's conditional FSDP/ABC compatibility context also remains; the normal Flash model path does not need that class replacement.

Qualification commands (CPU, no model downloads):

```sh
python -m pytest -q tests/runtime_interfaces --confcutdir=tests/runtime_interfaces
python -m pytest -q tests/workers/test_lora_bucket_accumulation_on_cpu.py --confcutdir=tests/workers
```

Use Transformers 5.12.1 and CPU PyTorch for the tiny dense/MoE forward/gradient/checkpoint tests. The optional `VERL_TEST_VLLM_WORKER_MANAGER=/path/to/vllm/v0.19.1/vllm/lora/worker_manager.py` check executes the real pinned manager classes with device/cache storage substituted, covering repeated adapter replacement and validation failure. Ordinary CPU tests still cover the service boundary without that source. No CPU test establishes CUDA kernel, FSDP, Ray, actual vLLM hook dispatch under compilation, or adapter-synchronization compatibility. Those remain required before these draft PRs are marked ready.
