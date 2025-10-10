"""Provide a minimal DeepSpeed shim when the real package is unavailable."""

from __future__ import annotations

import sys
import types
from typing import Any

import torch


class _TorchAccelerator:
    """Subset of the DeepSpeed accelerator interface used by the tests."""

    def __init__(self) -> None:
        self._update_device()

    def _update_device(self) -> None:
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            self._device = torch.device("cuda", index)
        else:
            self._device = torch.device("cpu")

    # Public API ---------------------------------------------------------
    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def device_count(self) -> int:
        return torch.cuda.device_count() if torch.cuda.is_available() else 1

    def set_device(self, device_id: int) -> None:
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            self._update_device()
        else:
            self._device = torch.device("cpu")

    def device_name(self) -> str:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(self._device.index or torch.cuda.current_device())
        return "cpu"

    def current_device_name(self) -> str:
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            return f"cuda:{index}"
        return "cpu"

    def current_device(self) -> int:
        if torch.cuda.is_available():
            return torch.cuda.current_device()
        return -1

    def communication_backend_name(self) -> str:
        return "nccl" if torch.cuda.is_available() else "gloo"

    def synchronize(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def get_op_builder(self, name: str):  # pragma: no cover - simple shim
        class _OpBuilderStub:
            def load(self):
                return None

        return _OpBuilderStub

    def manual_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def get_rng_state(self) -> torch.Tensor:
        if torch.cuda.is_available():
            return torch.cuda.get_rng_state()
        return torch.get_rng_state()

    def set_rng_state(self, state: torch.Tensor) -> None:
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(state)
        else:
            torch.set_rng_state(state)

    def lazy_call(self, cb):  # pragma: no cover - simple shim
        cb()

    def default_generator(self, index: int):
        if torch.cuda.is_available():
            return torch.cuda.default_generators[index]
        return torch.default_generator


_ACCELERATOR = _TorchAccelerator()


def _safe_get_full_fp32_param(param: Any, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - trivial shim
    """Fallback implementation that simply returns the parameter reference."""

    return param


def ensure_deepspeed_stub() -> None:
    """Install a lightweight DeepSpeed compatibility layer when necessary."""

    try:
        __import__("deepspeed")
        return
    except ModuleNotFoundError:
        pass

    module = types.ModuleType("deepspeed")
    module.__path__ = []  # type: ignore[attr-defined]
    accelerator_module = types.ModuleType("deepspeed.accelerator")
    accelerator_module.__path__ = []  # type: ignore[attr-defined]
    utils_module = types.ModuleType("deepspeed.utils")

    def _register_module(fullname: str) -> types.ModuleType:
        module = types.ModuleType(fullname)
        module.__path__ = []  # type: ignore[attr-defined]
        sys.modules.setdefault(fullname, module)
        return module

    runtime_module = _register_module("deepspeed.runtime")
    activation_module = _register_module("deepspeed.runtime.activation_checkpointing")
    checkpoint_module = _register_module(
        "deepspeed.runtime.activation_checkpointing.checkpointing"
    )
    zero_module = _register_module("deepspeed.runtime.zero")
    checkpointing_module = _register_module("deepspeed.checkpointing")
    real_accelerator_module = _register_module("deepspeed.accelerator.real_accelerator")
    moe_module = _register_module("deepspeed.moe")
    moe_layer_module = _register_module("deepspeed.moe.layer")
    pipe_module = _register_module("deepspeed.pipe")

    def get_accelerator() -> _TorchAccelerator:
        return _ACCELERATOR

    def _checkpoint(function: Any, *args: Any, **kwargs: Any) -> Any:
        return function(*args, **kwargs)

    class _GatheredParameters:
        def __init__(self, params: Any, *args: Any, **kwargs: Any) -> None:
            self._params = params

        def __enter__(self) -> Any:
            return self._params

        def __exit__(self, exc_type, exc, tb) -> bool:  # pragma: no cover - passthrough
            return False

    def _checkpointing_is_configured() -> bool:
        return False

    def _checkpointing_passthrough(*args: Any, **kwargs: Any) -> None:  # pragma: no cover - shim
        return None

    class _MoE(torch.nn.Module):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__()

        def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:  # type: ignore[override]
            return hidden_states

    class _LayerSpec:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

    class _TiedLayerSpec(_LayerSpec):
        pass

    class _PipelineModule(torch.nn.Module):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__()


    module.get_accelerator = get_accelerator  # type: ignore[attr-defined]
    module.accelerator = accelerator_module
    module.runtime = runtime_module  # type: ignore[attr-defined]
    module.moe = moe_module  # type: ignore[attr-defined]
    runtime_module.activation_checkpointing = activation_module  # type: ignore[attr-defined]
    runtime_module.zero = zero_module  # type: ignore[attr-defined]
    activation_module.checkpointing = checkpoint_module  # type: ignore[attr-defined]
    moe_module.layer = moe_layer_module  # type: ignore[attr-defined]
    module.checkpointing = checkpointing_module  # type: ignore[attr-defined]
    accelerator_module.get_accelerator = get_accelerator
    utils_module.safe_get_full_fp32_param = _safe_get_full_fp32_param
    real_accelerator_module.get_accelerator = get_accelerator
    checkpoint_module.checkpoint = _checkpoint
    zero_module.GatheredParameters = _GatheredParameters
    moe_layer_module.MoE = _MoE
    pipe_module.PipelineModule = _PipelineModule
    pipe_module.LayerSpec = _LayerSpec
    pipe_module.TiedLayerSpec = _TiedLayerSpec
    checkpointing_module.is_configured = _checkpointing_is_configured
    checkpointing_module.get_cuda_rng_tracker = _checkpointing_passthrough
    checkpointing_module.model_parallel_cuda_manual_seed = _checkpointing_passthrough
    checkpointing_module.model_parallel_reconfigure_tp_seed = _checkpointing_passthrough

    sys.modules.setdefault("deepspeed", module)
    sys.modules.setdefault("deepspeed.accelerator", accelerator_module)
    sys.modules.setdefault("deepspeed.utils", utils_module)


__all__ = ["ensure_deepspeed_stub"]
