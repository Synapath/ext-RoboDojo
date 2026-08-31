"""Serial, worker-private job mailbox. No socket, command execution or retries."""
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import re
import signal
import sys
import time
import traceback
import weakref

SCHEMA = "sim-service-resident-v2"
HEX = re.compile(r"^[0-9a-f]{32}$")
SPEC_FIELDS = {"task", "policy_adapter", "policy_host", "policy_port", "checkpoint_ref",
               "env_config", "action_type", "seed", "eval_num", "execution_horizon", "headless"}


def environment_mode(spec, previous_spec):
    if previous_spec is None:
        return "cold"
    return "rebuild"


def unsubscribe_context(simulation):
    """Pinned IsaacLab STOP and Isaac Sim PHYSICS_READY subscriptions."""
    if simulation is None:
        return
    simulation._disable_app_control_on_stop_handle = True
    handle = getattr(simulation, "_app_control_on_stop_handle", None)
    if handle is not None:
        handle.unsubscribe()
        simulation._app_control_on_stop_handle = None
    guard = getattr(simulation, "_on_post_physics_ready_callback", None)
    if guard is not None:
        guard.reset()
        simulation._on_post_physics_ready_callback = None


def clear_import_cache():
    # Kit caches OSError objects, whose traceback retains import caller frames
    # (including retired environments). CUDA empty_cache cannot release these.
    from omni.ext._impl import stat_cache
    stat_cache.reset_stat_cache()


def environment_refs(env):
    values = [("environment", env), ("robot_manager", env.robot_manager)]
    if env.sim is not None:
        values += [("isaac_environment", env.sim), ("physics_context", env.sim.sim)]
    values += [("planner", p) for p in env.robot_manager.planner.values()]
    values += [("camera", c) for c in env.capture_manager.tiled_cameras]
    if getattr(env, "stage", None) is not None:
        values.append(("stage", env.stage))
    return [(name, weakref.ref(obj)) for name, obj in values]


def collect_environment(refs):
    """Call only after ALL owning local variables have dropped the old env."""
    import torch
    clear_import_cache()
    gc.collect()
    torch.cuda.synchronize()
    alive = [name for name, ref in refs if ref() is not None]
    if alive:
        raise RuntimeError("Retired environment objects survived: " + ", ".join(alive))
    torch.cuda.empty_cache()
    return {"objects_released": len(refs), "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved()}


def retire_environment(env, app):
    """Release the pinned IsaacLab context without closing SimulationApp."""
    import omni.usd
    from omni.syntheticdata import SyntheticData
    from isaaclab.sim import SimulationContext
    from pxr import UsdUtils

    context = omni.usd.get_context()
    old_stage = context.get_stage_id()
    simulation = SimulationContext.instance()
    # IsaacLab inserts its initial stage in the global USD cache; closing the
    # context alone need not erase that strong reference. Keep exact owned
    # stages until close, then evict only these, never clear the whole cache.
    owned_stages = [context.get_stage(), getattr(env, "stage", None)]
    if simulation is not None:
        owned_stages.append(simulation.get_initial_stage())
    unsubscribe_context(simulation)
    products = len(env.capture_manager.tiled_render_products)
    # Tiled render products share SyntheticData dependencies. In 0.6.13,
    # per-annotator recursive detach can visit stale NodeObj entries. Retiring
    # the entire owned stage permits its public graph reset, which checks node
    # validity and clears activation history before individual camera release.
    SyntheticData.Get().reset()
    client = getattr(env, "model_client", None)
    if client is not None:
        client.close()
        env.model_client = None
    env.close()
    # RobotManager.close also serves native soft-reset callers. Permanent
    # destruction is service-only and must not change that upstream contract.
    manager = env.robot_manager
    errors, seen = [], set()
    for planner in (*manager.planner.values(), *manager.ik_solver.values()):
        if id(planner) in seen:
            continue
        seen.add(id(planner))
        try:
            planner.close()
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    if errors:
        raise RuntimeError("Planner destruction failed: " + "; ".join(errors))
    manager.planner.clear()
    manager.ik_solver.clear()
    manager.sim = manager.scene = None
    manager.robot_list.clear()
    if env.sim is not None or SimulationContext.instance() is not None:
        raise RuntimeError("Old physics context survived environment close")
    if (context.get_stage() is None or context.get_stage_id() == old_stage
            or not app.is_running()):
        raise RuntimeError("Environment close did not replace stage and retain application")
    stage_cache = UsdUtils.StageCache.Get()
    evicted = 0
    for stage in owned_stages:
        if stage is None:
            continue
        if stage == context.get_stage():
            raise RuntimeError("Retired stage is still current")
        if stage_cache.Contains(stage):
            stage_cache.Erase(stage)
            evicted += 1
        if stage_cache.Contains(stage):
            raise RuntimeError("Retired stage survived USD cache eviction")
    if env.capture_manager.tiled_render_products or env.capture_manager.tiled_cameras:
        raise RuntimeError("Old camera resources survived environment close")
    return {"old_stage_id": old_stage, "new_stage_id": context.get_stage_id(),
            "render_products_released": products, "physics_singleton_cleared": True,
            "syntheticdata_graphs_reset": True, "stage_cache_evictions": evicted}


class SessionShutdown(BaseException):
    """Unwind to the session owner, never close Kit inside a signal handler."""


class EnvironmentOwner:
    def __init__(self, app):
        self.app, self.env = app, None
        self.builds = 0
        self.closing = self.stopping = False
        self.retirements = []

    def signal(self, signum, frame):
        self.stopping = True
        if not self.closing:
            raise SessionShutdown(f"signal {signum}")

    def adopt(self, env):
        if self.env is not None:
            raise RuntimeError("Previous environment is still owned")
        self.env = env
        self.builds += 1

    def detach(self):
        if self.env is None:
            return []
        self.closing = True
        refs = environment_refs(self.env)
        checks = retire_environment(self.env, self.app)
        self.env = None
        self.retirements.append(checks)
        return refs

    def collect(self, refs, shutdown=False):
        checks = collect_environment(refs)
        if refs:
            self.retirements[-1].update(checks)
        self.closing = False
        if self.stopping and not shutdown:
            raise SessionShutdown("shutdown requested during cleanup")


def completed_progress(env):
    """Only scalar/result data crosses batch boundaries, never env objects."""
    if env.video_writers:
        raise RuntimeError("Completed batch still has open videos")
    return json.loads(json.dumps({
        "save_dir": env.save_dir, "success_nums": env.success_nums,
        "fail_nums": env.fail_nums, "total_score": env.total_score,
        "details": env.eval_result["details"],
        "abandoned_layout_ids": sorted(env.abandoned_seeds),
    }, allow_nan=False))


def reset_renderer_history(context):
    """Clear prior-frame accumulation without changing rendering settings.

    UsdContext.reset_renderer_accumulation is available in pinned omni.usd
    1.13.10. It must run after layout changes and before the native warmup.
    """
    context.reset_renderer_accumulation()


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
    if previous_spec is not None and spec["headless"] != previous_spec["headless"]:
        raise ValueError("Incompatible resident application")
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
    # Preserve the native PhysX monitor's pipes. Redirect their echo fds, not
    # stdout/stderr themselves (which would bypass monitoring during a job).
    monitor_module = sys.modules.get("src.eval_client.physx_warning_monitor")
    monitor = monitor_module.get_monitor() if monitor_module is not None else None
    targets = [monitor._saved_fd_by_target[k] for k in (1, 2)] if monitor is not None and monitor._started else [1, 2]
    saved = [os.dup(fd) for fd in targets]
    with path.open("xb", buffering=0) as stream:
        try:
            for target in targets:
                os.dup2(stream.fileno(), target)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            for target, original in zip(targets, saved):
                os.dup2(original, target)
                os.close(original)


def serve(args, run_job, app, profile):
    owner = EnvironmentOwner(app)
    previous_handlers = {sig: signal.signal(sig, owner.signal) for sig in (signal.SIGTERM, signal.SIGINT)}
    shutdown_error = None
    try:
        _serve_jobs(args, run_job, app, profile, owner)
    except SessionShutdown:
        pass
    finally:
        # No successful handoff is possible until this owner has released its
        # environment. Do not let repeated signals interrupt final cleanup.
        owner.closing = True
        try:
            refs = owner.detach()
            owner.collect(refs, shutdown=True)
        except BaseException as exc:
            shutdown_error = f"{type(exc).__name__}: {exc}"
        owner.closing = True
        from isaaclab.sim import SimulationContext
        # Includes construction failures before the env could be adopted.
        unsubscribe_context(SimulationContext.instance())
        atomic_json(Path(args.service_session) / "shutdown.json", {
            "signal_requested": owner.stopping, "cleanup_error": shutdown_error,
            "phase": "application-close", "environment_owned": owner.env is not None,
        })
        try:
            app.close()
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)


def _serve_jobs(args, run_job, app, profile, owner):
    root = Path(args.service_session)
    session_id = os.environ["SIM_SERVICE_SESSION_ID"]
    if not HEX.fullmatch(session_id):
        raise ValueError("Invalid session ID")
    sequence, previous_spec = 1, None
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
        if spec["policy_adapter"] == "Pi_05" and spec["execution_horizon"] is not None:
            os.environ["PI05_EXECUTION_HORIZON"] = str(spec["execution_horizon"])
        else:
            os.environ.pop("PI05_EXECUTION_HORIZON", None)
        args.task_name, args.env_cfg_type = spec["task"], spec["env_config"]
        args.policy_name, args.host, args.port = spec["policy_adapter"], spec["policy_host"], spec["policy_port"]
        args.policy_server_url = f"ws://{args.host}:{args.port}"
        args.seed = spec["seed"]
        args.additional_info = f"ckpt_name={spec['checkpoint_ref']},action_type={spec['action_type']}"
        profile.__init__(enabled=os.environ.get("ROBODOJO_PROFILE") == "1")
        mode = environment_mode(spec, previous_spec)
        owner.builds, owner.retirements = 0, []
        profile.metadata.update({"session_id": session_id, "sequence": sequence, "job_id": job_id,
                                 "warm": sequence > 1, "pid": os.getpid(), "application_id": id(app),
                                 "mode": mode, "environment_generation": sequence})
        if sequence == 1:
            profile.metadata["app_launch_s"] = launch_profile["elapsed_s"]
        receipt = {k: v for k, v in request.items() if k != "spec"}
        receipt.update({"pid": os.getpid(), "application_id": id(app), "warm": sequence > 1,
                        "mode": mode, "environment_generation": sequence, "code": 1, "error": None})
        failed = False
        with job_log(directory / "runner.log"):
            env = None
            try:
                profile.set_phase("configuration")
                env = run_job(resident=True, owner=owner)
                receipt.update({"environment_id": id(env), "simulator_id": id(env.sim),
                                "num_envs": env.num_envs, "seed": spec["seed"],
                                "eval_time": env.success_nums + env.fail_nums})
                profile.metadata.update({"environment_id": id(env), "simulator_id": id(env.sim)})
                profile.finished = True
            except BaseException as exc:
                traceback.print_exc()
                receipt["error"] = f"{type(exc).__name__}: {exc}"
                failed = True
            finally:
                env = None
            # The exception variable/traceback has left scope before GC.
            try:
                profile.set_phase("environment_teardown")
                refs = owner.detach()
                owner.collect(refs)
                receipt.update(environment_released=True, environment_builds=owner.builds)
                profile.metadata.update(environment_builds=owner.builds, teardown=owner.retirements)
            except BaseException as exc:
                traceback.print_exc()
                receipt["error"] = f"{receipt['error'] or ''} Cleanup: {type(exc).__name__}: {exc}"
                failed = True
            finally:
                receipt["code"] = 1 if failed else 0
                profile.write()
                profile.enabled = False  # atexit cannot mutate sealed job files
        atomic_json(root / f"completed-{sequence}.json", receipt)
        if failed:
            # A contaminated CUDA/scene context must never accept another job.
            return  # outer finally owns shutdown, no further requests/retries
        previous_spec = spec
        sequence += 1
