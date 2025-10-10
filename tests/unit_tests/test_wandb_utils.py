import pytest

from megatron import global_vars
from megatron.wandb_utils import (
    is_wandb_available,
    log_wandb_metrics,
    update_wandb_summary,
)


class DummyRun:
    def __init__(self):
        self.logged = []
        self.disabled = False
        self.summary = {}


class DummyWandB:
    def __init__(self):
        self.run = DummyRun()
        self.logged = []

    def log(self, data, **kwargs):
        self.logged.append((data, kwargs))

    def finish(self):  # pragma: no cover - compatibility shim
        self.run = None


@pytest.fixture(autouse=True)
def clear_wandb_writer():
    original = global_vars._GLOBAL_WANDB_WRITER
    global_vars._GLOBAL_WANDB_WRITER = None
    try:
        yield
    finally:
        global_vars._GLOBAL_WANDB_WRITER = original


def test_is_wandb_available_without_writer():
    global_vars._GLOBAL_WANDB_WRITER = None
    assert not is_wandb_available()
    # Should not raise even though there is no writer.
    log_wandb_metrics({"metric": 1})


def test_log_wandb_metrics_forwards_to_active_run():
    dummy = DummyWandB()
    global_vars._GLOBAL_WANDB_WRITER = dummy

    log_wandb_metrics({"metric": 1}, step=7, commit=False)

    assert dummy.logged == [({"metric": 1}, {"step": 7, "commit": False})]
    assert is_wandb_available()

    update_wandb_summary({"best": 5})
    assert dummy.run.summary["best"] == 5
