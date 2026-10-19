# Explicit trainer interfaces (fork API version 1)

This fork exposes opt-in construction points for packaged training recipes. `verl.runtime_interfaces.API_VERSION` identifies this contract. Default classes and configuration preserve the existing behavior; recipes do not replace module globals or dependency methods.

- `HFModelConfig.custom_engine_class` is an importable `BaseEngine` subclass. `runtime_options` is a recipe-owned serializable dictionary. `TrainingWorker` resolves the class for each worker; it does not modify `EngineRegistry`. The recipe must select an engine compatible with its configured model and backend.
- `TaskRunner.trainer_class` selects the GRPO/PPO trainer. `SFTTrainer.tracking_class`, `SFTTrainer.checkpoint_handler_class`, and both PPO trainers' `tracking_class` select services with the original constructor signatures. The recipe owns their lifetime.
- `RayPPOTrainer._compute_advantage` and `_apply_rollout_correction` delegate to the original functions by default. Overrides must preserve batch structure and return values.
- `PPOTrainer._create_replay_buffer` constructs a buffer owned by that trainer. The caller remains responsible for closing it.
- `AgentLoopWorker._create_teacher_manager` selects a teacher service. `generate_sequences(..., trajectory_info=...)` accepts a per-row identity sidecar; its length must match the batch. Omitting it retains the existing identity generation.
- `AgentLoopWorkerTQBase` is an ordinary subclassable worker. `AgentLoopWorkerTQ` remains its Ray actor wrapper. `AgentLoopManagerTQ.worker_class` selects that wrapper per manager type.

LoRA rollout now fails before generation when the configured adapter has not been loaded. Previously that condition silently generated from the base model. Full-model rollouts remain unchanged.

These interfaces do not remove Verl's internal model compatibility code. Consumers must pin and qualify this fork version with their supported CUDA, PyTorch, Transformers, and vLLM stack. CPU contract tests exercise construction and fail-closed behavior; distributed training and actual adapter synchronization require GPU qualification.
