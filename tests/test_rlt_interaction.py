import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("rlt_interaction", Path(__file__).parents[1] / "utils/rlt_interaction.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def env(monkeypatch):
    monkeypatch.setenv("ROBODOJO_RLT_INTERACTION", "1")
    return SimpleNamespace(
        task_name="plug_in_charger",
        num_envs=1,
        charger_diagnostics_enabled=True,
        take_action_cnt=[2],
        end_flag=[True],
        success=[True],
        step_lim=400,
        rlt_observations={0: (2, {"state": {"q": [1]}})},
        rlt_applied_actions={0: {"q": [2]}},
        charger_diagnostics={0: SimpleNamespace(status="complete", rows=[{"action_count": 2, "time_seconds": 0.08}])},
    )


def test_terminal_read_is_cached_and_cannot_step(monkeypatch):
    e = env(monkeypatch)
    r = m.read_frame(e)
    assert r["native_success"] and not r["time_limit"]
    r["observation"]["state"]["q"][0] = 9
    assert e.rlt_observations[0][1]["state"]["q"] == [1]
    with pytest.raises(ValueError):
        m.step(e, {})


def test_missing_terminal_not_rerendered(monkeypatch):
    e = env(monkeypatch)
    e.rlt_observations = {}
    with pytest.raises(RuntimeError):
        m.read_frame(e)


def test_timeout_and_error_not_success(monkeypatch):
    e = env(monkeypatch)
    e.success = [False]
    assert m.read_frame(e)["error"]
    e.step_lim = 2
    r = m.read_frame(e)
    assert r["time_limit"] and not r["native_success"] and not r["error"]


def test_scope_opt_in(monkeypatch):
    e = env(monkeypatch)
    e.num_envs = 2
    with pytest.raises(ValueError):
        m.read_frame(e)
    e.num_envs = 1
    monkeypatch.delenv("ROBODOJO_RLT_INTERACTION")
    with pytest.raises(ValueError):
        m.read_frame(e)
