"""Utilities for interacting with Weights & Biases logging.

These helpers centralise the logic that checks whether a W&B run is
active before attempting to emit metrics.  Historically the project made
ad-hoc calls to :mod:`wandb` scattered across several modules, which made
it easy to forget the guard rails required for CPU-only runs or
configurations where W&B is not installed.  Consolidating the logic in a
single place lets the rest of the codebase focus on assembling metrics
without worrying about the underlying logging transport.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, MutableMapping, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for static type checkers only
    from megatron.global_vars import get_wandb_writer


def _resolve_wandb() -> tuple[Optional[Any], Optional[Any]]:
    """Return the active W&B module and its log callable if available.

    The helper mirrors the ``get_wandb_writer`` semantics where the
    returned object may either be ``None`` (when W&B is disabled) or the
    imported :mod:`wandb` module.  The caller still needs to make sure
    that an experiment has been ``wandb.init``'d which we expose via the
    presence of ``writer.run``.
    """

    from megatron import global_vars

    writer = global_vars.get_wandb_writer()
    if writer is None:
        return None, None

    log_callable = getattr(writer, "log", None)
    if not callable(log_callable):
        return writer, None

    return writer, log_callable


def is_wandb_available() -> bool:
    """Return ``True`` when a W&B run is active on the current process."""

    writer, log_callable = _resolve_wandb()
    if writer is None or log_callable is None:
        return False

    run = getattr(writer, "run", None)
    if run is None:
        return False

    if getattr(run, "disabled", False):
        return False

    return True


def log_wandb_metrics(
    metrics: Mapping[str, Any],
    *,
    step: Optional[int] = None,
    commit: Optional[bool] = None,
) -> None:
    """Log ``metrics`` to the active W&B run if one is available.

    Parameters
    ----------
    metrics:
        Mapping of metric names to values.  The mapping is copied so the
        caller can reuse the input dictionary.
    step:
        Optional explicit global step for the update.  When omitted W&B
        follows its default behaviour which increments the step based on
        previous calls.
    commit:
        Optional flag forwarded to :func:`wandb.log` controlling whether
        the update should finish the current history item.
    """

    if not metrics:
        return

    writer, log_callable = _resolve_wandb()
    if writer is None or log_callable is None:
        return

    run = getattr(writer, "run", None)
    if run is None or getattr(run, "disabled", False):
        return

    log_kwargs: MutableMapping[str, Any] = {}
    if step is not None:
        log_kwargs["step"] = step
    if commit is not None:
        log_kwargs["commit"] = commit

    # ``wandb.log`` mutates the input mapping when ``commit`` is provided,
    # hence we pass a shallow copy to keep the helper side-effect free.
    log_callable(dict(metrics), **log_kwargs)


def update_wandb_summary(values: Mapping[str, Any]) -> None:
    """Update the W&B run summary with the provided values.

    This is a thin convenience wrapper that mirrors ``wandb.run.summary``
    semantics while gracefully handling runs that are disabled or not
    initialised.
    """

    if not values:
        return

    writer, _ = _resolve_wandb()
    if writer is None:
        return

    run = getattr(writer, "run", None)
    if run is None or getattr(run, "disabled", False):
        return

    summary = getattr(run, "summary", None)
    if summary is None:
        return

    summary.update(dict(values))


def finish_wandb_run() -> None:
    """Finish the active W&B run if one exists."""

    writer, _ = _resolve_wandb()
    if writer is None:
        return

    finish = getattr(writer, "finish", None)
    if callable(finish):
        finish()
