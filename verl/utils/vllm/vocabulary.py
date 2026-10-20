"""Tokenizer vocabulary limits owned by each constructed rollout model."""

from functools import partial


def _mask_unavailable_tokens(_module, _inputs, logits, *, vocab_size):
    # Non-output tensor-parallel ranks can return None.
    if logits is not None:
        logits[..., vocab_size:] = float("-inf")
    return logits


def configure_model_vocabulary(model, vocab_size: int):
    """Register PyTorch output hooks, preserving native compute_logits methods.

    Hook the outer LoRA logits processor where present, after its adapter delta
    and TP gather. Mask before sampling AND log-probability normalization, so
    raw/processed logprobs and speculative decoding retain their old semantics.
    The caller owns and removes handles before reconfiguration or teardown.
    """
    from vllm.lora.layers.logits_processor import LogitsProcessorWithLoRA
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("tokenizer vocabulary size must be a positive integer")
    targets = []
    for name, module in model.named_modules():
        if any(not parent or name.startswith(parent + ".") for parent, _ in targets):
            continue
        if isinstance(module, (LogitsProcessor, LogitsProcessorWithLoRA)):
            targets.append((name, module))
    if not targets:
        raise RuntimeError("rollout model has no supported logits processor")
    return [
        module.register_forward_hook(partial(_mask_unavailable_tokens, vocab_size=vocab_size)) for _, module in targets
    ]
