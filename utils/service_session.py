"""Serial, worker-private job mailbox. No socket, command execution or retries."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

SCHEMA = "sim-service-resident-v1"
HEX = re.compile(r"^[0-9a-f]{32}$")
SPEC_FIELDS = {"task", "policy_adapter", "policy_host", "policy_port", "checkpoint_ref",
               "env_config", "action_type", "seed", "eval_num", "execution_horizon", "headless"}
DYNAMIC_FIELDS = {"policy_host", "policy_port", "checkpoint_ref", "seed"}


def validate_request(value, session_id, sequence, previous_spec=None):
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "session_id", "sequence", "job_id", "request_hash", "nonce", "spec"
    }:
        raise ValueError("Invalid resident request fields")
    if value["schema_version"] != SCHEMA or value["session_id"] != session_id:
        raise ValueError("Resident session identity mismatch")
    if type(value["sequence"]) is not int or value["sequence"] != sequence:
        raise ValueError("Resident sequence mismatch")
    for name in ("job_id", "nonce"):
        if not isinstance(value[name], str) or not HEX.fullmatch(value[name]):
            raise ValueError("Invalid resident identity")
    if not isinstance(value["request_hash"], str) or not re.fullmatch(r"[0-9a-f]{64}", value["request_hash"]):
        raise ValueError("Invalid request hash")
    spec = value["spec"]
    if not isinstance(spec, dict) or set(spec) != SPEC_FIELDS or spec["headless"] is not True:
        raise ValueError("Invalid resident spec")
    if previous_spec is not None and any(spec[k] != previous_spec[k] for k in SPEC_FIELDS - DYNAMIC_FIELDS):
        raise ValueError("Incompatible resident environment")
    return spec


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def job_log(path):
    """Capture Python and native writes; restore fds before completion ACK."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    with path.open("xb", buffering=0) as stream:
        try:
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            for target, original in zip((1, 2), saved):
                os.dup2(original, target)
                os.close(original)


def serve(args, run_job, app, profile):
    root = Path(args.service_session)
    session_id = os.environ["SIM_SERVICE_SESSION_ID"]
    if not HEX.fullmatch(session_id):
        raise ValueError("Invalid session ID")
    sequence, env, previous_spec = 1, None, None
    # App startup has already occurred: preserve its first-job profiling cost.
    launch_profile = profile.snapshot()
    profile.enabled = False  # idle time is never charged to a job
    while True:
        inbox = root / "request.json"
        if not inbox.exists():
            time.sleep(0.05)
            continue
        if inbox.is_symlink() or inbox.stat().st_size > 65536:
            raise ValueError("Invalid resident inbox")
        request = json.loads(inbox.read_text())
        if request.get("sequence", 0) < sequence:
            time.sleep(0.05)
            continue
        spec = validate_request(request, session_id, sequence, previous_spec)
        job_id = request["job_id"]
        directory = root / "jobs" / job_id
        directory.mkdir(parents=True, exist_ok=False)
        output = Path("eval_result") / job_id
        output.mkdir(exist_ok=False)
        os.environ["ROBODOJO_RUN_ID"] = "service-" + job_id
        os.environ["ROBODOJO_OUTPUT_ROOT"] = str(output)
        os.environ["EVAL_NUM"] = str(spec["eval_num"])
        args.task_name, args.env_cfg_type = spec["task"], spec["env_config"]
        args.policy_name, args.host, args.port = spec["policy_adapter"], spec["policy_host"], spec["policy_port"]
        args.policy_server_url = f"ws://{args.host}:{args.port}"
        args.seed = spec["seed"]
        args.additional_info = f"ckpt_name={spec['checkpoint_ref']},action_type={spec['action_type']}"
        profile.__init__(enabled=os.environ.get("ROBODOJO_PROFILE") == "1")
        profile.metadata.update({"session_id": session_id, "sequence": sequence, "job_id": job_id,
                                 "warm": env is not None, "pid": os.getpid()})
        if sequence == 1:
            profile.metadata["app_launch_s"] = launch_profile["elapsed_s"]
        receipt = {k: v for k, v in request.items() if k != "spec"}
        receipt.update({"pid": os.getpid(), "warm": env is not None, "code": 1, "error": None})
        failed = False
        with job_log(directory / "runner.log"):
            try:
                env = run_job(env=env, resident=True)
                receipt.update({"code": 0, "environment_id": id(env), "simulator_id": id(env.sim),
                                "num_envs": env.num_envs, "seed": spec["seed"],
                                "eval_time": env.success_nums + env.fail_nums})
                profile.metadata.update({"environment_id": id(env), "simulator_id": id(env.sim)})
                profile.finished = True
            except BaseException as exc:
                traceback.print_exc()
                receipt["error"] = f"{type(exc).__name__}: {exc}"
                failed = True
            finally:
                profile.write()
                profile.enabled = False  # atexit cannot mutate sealed job files
        atomic_json(root / f"completed-{sequence}.json", receipt)
        if failed:
            # A contaminated CUDA/scene context must never accept another job.
            app.close()
            raise RuntimeError("Resident job failed; session terminated")
        previous_spec = spec
        sequence += 1
