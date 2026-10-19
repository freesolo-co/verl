"""Versioned, opt-in interfaces for explicitly constructed trainer recipes.

Default recipes retain the registry-selected engine and the standard trainer services.
See docs/runtime_interfaces.md for ownership and compatibility guarantees.
"""

API_VERSION = 1


def resolve_engine_class(model_config, model_type: str, backend: str):
    """Resolve one engine without changing the process-global engine registry."""
    from verl.utils.import_utils import load_class_from_fqn
    from verl.workers.engine import BaseEngine, EngineRegistry

    custom = model_config.get("custom_engine_class")
    if not custom:
        return EngineRegistry.get_engine_cls(model_type, backend)
    engine_cls = load_class_from_fqn(custom, "training engine")
    if not isinstance(engine_cls, type) or not issubclass(engine_cls, BaseEngine):
        raise TypeError("custom_engine_class must inherit BaseEngine")
    return engine_cls
