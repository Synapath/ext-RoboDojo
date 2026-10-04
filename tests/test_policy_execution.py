"""CPU tests; integration uses the pinned image's real XPolicyLab eval loops."""

import ast
import builtins
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from utils.policy_execution import execution_client, validate_horizon
from utils.service_session import validate_request
import test_service_session as session_tests

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("value", [None, 0, -1, True, False, "16", 16.0])
def test_invalid_horizon_rejected_at_mailbox(value):
    request = session_tests.ResidentContracts().request()
    request["spec"]["execution_horizon"] = value
    with pytest.raises(ValueError, match="positive integer"):
        validate_request(request, "a" * 32, 1)


@pytest.mark.parametrize("value", [1, 7, 16, 20, 50, 1024, 2**70])
def test_arbitrary_horizon_accepted_at_mailbox(value):
    request = session_tests.ResidentContracts().request()
    request["spec"].update(action_type="joint", execution_horizon=value)
    assert validate_request(request, "a" * 32, 1)["execution_horizon"] == value


def test_native_passthrough_and_per_episode_horizon(monkeypatch):
    client = SimpleNamespace(call=lambda **kw: list(range(50)))
    monkeypatch.delenv("PI05_EXECUTION_HORIZON", raising=False)
    monkeypatch.delenv("SIM_SERVICE_SESSION_ID", raising=False)
    assert execution_client("Pi_05", client) is client
    for value in (16, 7, 50):
        monkeypatch.setenv("PI05_EXECUTION_HORIZON", str(value))
        assert len(execution_client("Pi_05", client).call("get_action")) == value
        assert execution_client("OpenVLA", client) is client
    monkeypatch.delenv("PI05_EXECUTION_HORIZON")
    monkeypatch.setenv("SIM_SERVICE_SESSION_ID", "a" * 32)
    with pytest.raises(ValueError, match="positive integer"):
        execution_client("Pi_05", client)
    validate_horizon("OpenVLA", None)
    with pytest.raises(ValueError, match="only supported"):
        validate_horizon("OpenVLA", 16)


@pytest.mark.parametrize("raw", ["", "0", "-1", "true", "16.0", " 16", "１６"])
def test_invalid_environment_rejected(monkeypatch, raw):
    monkeypatch.setenv("PI05_EXECUTION_HORIZON", raw)
    with pytest.raises(ValueError, match="positive integer"):
        execution_client("Pi_05", object())


@pytest.mark.parametrize("result", [[], None, {}, "bad", [[]], [[1] * 20, [2] * 21]])
def test_bad_batch_rejected_before_upstream_execution(monkeypatch, result):
    monkeypatch.setenv("PI05_EXECUTION_HORIZON", "16")
    client = execution_client("Pi_05", SimpleNamespace(call=lambda **kw: result))
    with pytest.raises(ValueError):
        client.call("get_action_batch")


def test_passthrough_errors_and_nonmutating_array_slice(monkeypatch):
    monkeypatch.setenv("PI05_EXECUTION_HORIZON", "16")
    actions = np.arange(50 * 14).reshape(50, 14)
    client = execution_client("Pi_05", SimpleNamespace(call=lambda **kw: actions))
    np.testing.assert_array_equal(client.call("get_action"), actions[:16])
    assert actions.shape == (50, 14)
    assert client.call("reset") is actions
    def fail(**kwargs):
        raise TimeoutError("policy unavailable")
    client = execution_client("Pi_05", SimpleNamespace(call=fail))
    with pytest.raises(TimeoutError, match="unavailable"):
        client.call("get_action")


def test_batch_size_matches_active_environments(monkeypatch):
    monkeypatch.setenv("PI05_EXECUTION_HORIZON", "16")
    client = execution_client("Pi_05", SimpleNamespace(call=lambda **kw: [[{}] * 50]))
    with pytest.raises(ValueError, match="active environments"):
        client.call("get_action_batch", obs=[0, 1])


def test_native_joint_keys_dimensions_and_action_selection():
    validate = session_tests.load_method("src/eval_client/eval_env.py", "EvalEnv",
                                        "validate_action_dict", {"np": np, "deepcopy": deepcopy})
    select = session_tests.load_method("src/eval_client/eval_env.py", "EvalEnv", "get_action_type", {})
    env = SimpleNamespace(robot_action_dim_info={"arm_dim": [6, 6], "ee_dim": [1, 1]},
                          robot_manager=SimpleNamespace(
                              robot_list=[SimpleNamespace(type="target", arm_name=f"{side}_arm")
                                          for side in ("left", "right")],
                              process_name=lambda name: name + "_joint_state"))
    action = {f"{side}_{part}_joint_state": [0.] * size
              for side in ("left", "right") for part, size in (("arm", 6), ("ee", 1))}
    validate(env, action)
    assert select(env, action) == "joint"
    action["left_arm_joint_state"] = [0.] * 7
    with pytest.raises(ValueError, match="dim mismatch"):
        validate(env, action)


@pytest.fixture
def invoke():
    path = ROOT / "XPolicyLab/policy/Pi_05/deploy.py"
    if not path.exists():
        pytest.skip("Pinned runtime image is required for official-loop integration")
    spec = importlib.util.spec_from_file_location("pinned_pi05_deploy", path)
    deploy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deploy)
    # Exercise our actual EvalEnv entry points without importing/starting Isaac.
    tree = ast.parse((ROOT / "src/eval_client/eval_env.py").read_text())
    def run(env, mode):
        name = "eval_one_episode" + ("_batch" if mode == "batch" else "")
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        def import_deploy(name, *args, **kwargs):
            assert name == "XPolicyLab.policy.Pi_05.deploy"
            return deploy
        scope = {"execution_client": execution_client,
                 "__builtins__": {**vars(builtins), "__import__": import_deploy}}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
        scope[name](env)
    return run


class FakeEnv:
    def __init__(self, limits):
        self.limits = limits
        self.actions = [[] for _ in limits]
        self.deploy_cfg = {"policy_name": "Pi_05"}

    def is_episode_end(self):
        return all(len(actions) == limit for actions, limit in zip(self.actions, self.limits))

    def get_running_env_idx_list(self):
        return [i for i, (actions, limit) in enumerate(zip(self.actions, self.limits)) if len(actions) < limit]

    def get_obs_batch(self, indices):
        return [{"env_idx": i, "step": len(self.actions[i])} for i in indices]

    def get_obs(self):
        return self.get_obs_batch([0])[0]

    def take_action_batch(self, actions, indices):
        assert len(actions) == len(indices)
        for action, i in zip(actions, indices):
            assert len(self.actions[i]) < self.limits[i]
            self.actions[i].append(action)

    def take_action(self, action):
        self.take_action_batch([action], [0])


class FakePolicy:
    def __init__(self, chunk_size):
        self.chunk_size = chunk_size
        self.inferences, self.responses, self.observations = [], [], []

    def call(self, func_name, **kwargs):
        if func_name == "update_obs_batch":
            self.observations = kwargs["obs"]
        elif func_name == "update_obs":
            self.observations = [kwargs["obs"]]
        elif func_name in ("get_action", "get_action_batch"):
            generation = len(self.inferences)
            self.inferences.append({o["env_idx"]: o["step"] for o in self.observations})
            actions = [[{"generation": generation, "offset": j} for j in range(self.chunk_size)]
                       for _ in self.observations]
            self.responses.append(actions)
            return actions[0] if func_name == "get_action" else actions


@pytest.mark.parametrize("mode", ["single", "batch"])
@pytest.mark.parametrize("horizon,chunk_size", [(1, 50), (7, 50), (16, 50), (20, 50),
                                             (50, 50), (100, 50), (16, 3), (2**70, 50)])
def test_actual_entrypoint_reinfers_and_discards_tail(invoke, monkeypatch, mode, horizon, chunk_size):
    monkeypatch.setenv("PI05_EXECUTION_HORIZON", str(horizon))
    env, policy = FakeEnv([53]), FakePolicy(chunk_size)
    env.model_client = policy
    invoke(env, mode)
    size = min(horizon, chunk_size)
    assert policy.inferences == [{0: step} for step in range(0, 53, size)]
    assert len(policy.inferences) == math.ceil(53 / size)
    assert env.actions[0] == [{"generation": i // size, "offset": i % size} for i in range(53)]
    assert all(len(chunk) == chunk_size for batch in policy.responses for chunk in batch)
    assert env.model_client is policy  # No persistent wrapper retaining prior jobs.


def test_batch_shrinks_when_environment_finishes(invoke, monkeypatch):
    monkeypatch.setenv("PI05_EXECUTION_HORIZON", "16")
    env, policy = FakeEnv([10, 35]), FakePolicy(50)
    env.model_client = policy
    invoke(env, "batch")
    assert list(map(len, env.actions)) == [10, 35]
    assert policy.inferences == [{0: 0, 1: 0}, {1: 16}, {1: 32}]


@pytest.mark.parametrize("mode", ["single", "batch"])
def test_episode_can_end_before_horizon(invoke, monkeypatch, mode):
    monkeypatch.setenv("PI05_EXECUTION_HORIZON", "16")
    env, policy = FakeEnv([3]), FakePolicy(50)
    env.model_client = policy
    invoke(env, mode)
    assert policy.inferences == [{0: 0}] and len(env.actions[0]) == 3
