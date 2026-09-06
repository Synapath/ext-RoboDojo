"""Opt-in single-environment RLT access to already captured native observations."""

from copy import deepcopy


def enabled(env):
    import os

    active = os.environ.get("ROBODOJO_RLT_INTERACTION") == "1"
    if active and (env.task_name != "plug_in_charger" or env.num_envs != 1 or not env.charger_diagnostics_enabled):
        raise ValueError("RLT requires charger, one environment and diagnostics")
    return active


def read_frame(env):
    if not enabled(env):
        raise ValueError("RLT interaction is not enabled")
    count = int(env.take_action_cnt[0])
    cached = getattr(env, "rlt_observations", {}).get(0)
    if cached is None or cached[0] != count:
        if env.end_flag[0]:
            raise RuntimeError("native endpoint observation missing")
        env.get_obs_batch([0])
        cached = env.rlt_observations[0]
    diag = env.charger_diagnostics.get(0)
    if diag is None or diag.status != "complete" or not diag.rows or diag.rows[-1]["action_count"] != count:
        raise RuntimeError("current diagnostic observation missing or invalid")
    row = deepcopy(diag.rows[-1])
    ended = bool(env.end_flag[0])
    success = ended and bool(env.success[0])
    timeout = ended and not success and count >= env.step_lim
    error = "native non-timeout failure" if ended and not success and not timeout else None
    return {
        "step": count,
        "time_s": row["time_seconds"],
        "observation": deepcopy(cached[1]),
        "native_success": success,
        "time_limit": timeout,
        "error": error,
        "diagnostics": row,
        "applied_dict": deepcopy(getattr(env, "rlt_applied_actions", {}).get(0)),
    }


def step(env, command):
    if not enabled(env):
        raise ValueError("RLT interaction is not enabled")
    if env.end_flag[0]:
        raise ValueError("step after native endpoint")
    previous = int(env.take_action_cnt[0])
    env.take_action(command)
    if env.take_action_cnt[0] != previous + 1:
        raise RuntimeError("command was not executed")
    # take_action owns reward progression and captures the last frame on terminal.
    return read_frame(env)
