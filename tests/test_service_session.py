"""CPU-only contracts for resident transport and evaluation-state reset."""
import ast
from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
import sys
import gc
import json
import tempfile
import weakref
from unittest.mock import patch

from utils.service_session import (validate_request, environment_mode, retire_environment,
    EnvironmentOwner, SessionShutdown, collect_environment, completed_progress, _serve_jobs, serve)
from utils.performance import WallProfile

ROOT = Path(__file__).resolve().parents[1]


def load_method(relative, class_name, method_name, namespace):
    # Execute the actual method without importing Isaac or launching an app.
    module = ast.parse((ROOT / relative).read_text())
    cls = next(node for node in ast.walk(module) if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(relative), "exec"), namespace)
    return namespace[method_name]


class DummyManager:
    def __init__(self, *args, **kwargs):
        self.closed = False
    def initialize(self, env):
        self.env = env
    def init_eval(self, **kwargs):
        self.completed = kwargs
    def close(self):
        self.closed = True
    def call(self, *args, **kwargs):
        pass


class ResidentContracts(unittest.TestCase):
    def request(self):
        return {"schema_version": "sim-service-resident-v2", "session_id": "a" * 32,
                "sequence": 1, "job_id": "b" * 32, "nonce": "c" * 32, "request_hash": "d" * 64,
                "spec": {"task": "stack_bowls", "policy_adapter": "Pi_05", "policy_host": "10.0.0.2",
                         "policy_port": 9999, "checkpoint_ref": "hold", "env_config": "arx_x5",
                         "action_type": "ee", "seed": 0, "eval_num": 1,
                         "execution_horizon": 20, "headless": True}}

    def test_mailbox_rejects_stale_identity_and_extra_paths(self):
        value = self.request()
        self.assertEqual(validate_request(value, "a" * 32, 1), value["spec"])
        for key, bad in (("sequence", 0), ("session_id", "b" * 32), ("job_id", "../escape"),
                         ("nonce", "bad"), ("request_hash", "bad")):
            changed = deepcopy(value)
            changed[key] = bad
            with self.assertRaises(ValueError):
                validate_request(changed, "a" * 32, 1)
        value["output_path"] = "/history"
        with self.assertRaises(ValueError):
            validate_request(value, "a" * 32, 1)

    def test_all_subsequent_jobs_rebuild_even_identical_requests(self):
        value = self.request()
        previous = deepcopy(value["spec"])
        value["spec"].update(seed=1, policy_host="10.0.0.3", policy_port=8000, checkpoint_ref="next")
        validate_request(value, "a" * 32, 1, previous)
        self.assertEqual(environment_mode(value["spec"], previous), "rebuild")
        self.assertEqual(environment_mode(previous, previous), "rebuild")
        self.assertEqual(environment_mode(value["spec"], None), "cold")
        for key, bad in (("task", "insert_gear"), ("eval_num", 2), ("execution_horizon", 50)):
            changed = deepcopy(value)
            changed["spec"][key] = bad
            validate_request(changed, "a" * 32, 1, previous)
            self.assertEqual(environment_mode(changed["spec"], previous), "rebuild")

    def test_camera_cleanup_detaches_actual_annotators_once(self):
        cleanup = load_method("env/camera_manager/capture/camera_view.py", "CameraView", "_clean_up_tiled_sensor", {})
        events = []
        camera = SimpleNamespace(_render_product=SimpleNamespace(destroy=lambda: events.append("destroy")),
            _render_product_path="/Render/test", _annotators={"rgb": SimpleNamespace(detach=lambda paths: events.append(paths))})
        cleanup(camera)
        cleanup(camera)
        self.assertEqual(events, [["/Render/test"], "destroy"])
        self.assertEqual(camera._annotators, {})

    def test_retire_unsubscribes_before_close_and_checks_new_stage(self):
        events = []
        context = SimpleNamespace(get_stage_id=lambda: stage[0], get_stage=lambda: stage[0])
        usd = SimpleNamespace(get_context=lambda: context)
        singleton, stage = [None], [1]
        simulation = SimpleNamespace(_app_control_on_stop_handle=SimpleNamespace(unsubscribe=lambda: events.append("unsubscribe")))
        singleton[0] = simulation
        simulation._on_post_physics_ready_callback = SimpleNamespace(reset=lambda: events.append("post-ready-reset"))
        simulation.get_initial_stage = lambda: 1
        cache = {1, 2, 99}  # current replacement and unrelated stages must survive
        planner = SimpleNamespace(close=lambda: events.append("planner-close"))
        env = SimpleNamespace(sim=SimpleNamespace(sim=simulation), capture_manager=SimpleNamespace(tiled_render_products=[1, 2, 3], tiled_cameras=[1]),
            robot_manager=SimpleNamespace(planner={"x5":planner}, ik_solver={"x5":planner}, robot_list=[1]))
        def close():
            self.assertIsNone(simulation._app_control_on_stop_handle)
            events.append("close")
            env.sim = None
            singleton[0] = None
            stage[0] = 2
            env.capture_manager.tiled_render_products.clear()
            env.capture_manager.tiled_cameras.clear()
        env.close = close
        modules = {"omni": SimpleNamespace(usd=usd), "omni.usd": usd,
            "omni.syntheticdata": SimpleNamespace(SyntheticData=SimpleNamespace(Get=lambda: SimpleNamespace(reset=lambda: events.append("graph-reset")))),
            "isaaclab.sim": SimpleNamespace(SimulationContext=SimpleNamespace(instance=lambda: singleton[0])),
            "pxr": SimpleNamespace(UsdUtils=SimpleNamespace(StageCache=SimpleNamespace(Get=lambda:
                SimpleNamespace(Contains=lambda stage: stage in cache, Erase=cache.remove))))}
        with patch.dict(sys.modules, modules):
            result = retire_environment(env, SimpleNamespace(is_running=lambda: True))
        self.assertEqual(events, ["unsubscribe", "post-ready-reset", "graph-reset", "close", "planner-close"])
        self.assertEqual(result["render_products_released"], 3)
        self.assertEqual(result["new_stage_id"], 2)
        self.assertEqual(result["stage_cache_evictions"], 1)
        self.assertEqual(cache, {2, 99})

    def test_planner_explicit_close_is_idempotent_and_releases_all_fields(self):
        close = load_method("env/planner_manager/curobo_planner.py", "CuroboPlanner", "close", {})
        events = []
        planner = DummyManager()
        for name in ("ik_solver", "motion_planner", "motion_planner_batch"):
            setattr(planner, name, SimpleNamespace(destroy=lambda name=name: events.append(name)))
        planner.large_tensor = object()
        close(planner)
        close(planner)
        self.assertEqual(len(events), 3)
        self.assertEqual(planner.__dict__, {"_closed":True})

    def test_planner_close_attempts_all_resources_and_reports_failure(self):
        close = load_method("env/planner_manager/curobo_planner.py", "CuroboPlanner", "close", {})
        events = []
        def fail():
            raise RuntimeError("graph teardown failed")
        planner = SimpleNamespace(ik_solver=SimpleNamespace(destroy=fail),
            motion_planner=SimpleNamespace(destroy=lambda:events.append("motion")),
            motion_planner_batch=SimpleNamespace(destroy=lambda:events.append("batch")))
        with self.assertRaisesRegex(RuntimeError, "graph teardown failed"):
            close(planner)
        self.assertEqual(events, ["motion", "batch"])

    def test_gc_requires_dead_owners_not_just_free_allocator_cache(self):
        marker = DummyManager()
        ref = weakref.ref(marker)
        calls = []
        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None,
            empty_cache=lambda: calls.append("empty"), memory_allocated=lambda:0, memory_reserved=lambda:0))
        with patch.dict(sys.modules, {"torch":torch}), patch("utils.service_session.clear_import_cache") as clear:
            with self.assertRaisesRegex(RuntimeError, "environment"):
                collect_environment([("environment", ref)])
            self.assertEqual(calls, [])
            del marker
            collect_environment([("environment", ref)])
            self.assertEqual(calls, ["empty"])
            self.assertEqual(clear.call_count, 2)

    def test_import_exception_cache_is_cleared_before_owner_gc(self):
        cached = []
        def make():
            env = DummyManager()
            try:
                raise FileNotFoundError("import probe")
            except FileNotFoundError as exc:
                cached.append(exc)
            return weakref.ref(env)
        ref = make()
        gc.collect()
        self.assertIsNotNone(ref())
        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:None, empty_cache=lambda:None,
            memory_allocated=lambda:0, memory_reserved=lambda:0))
        modules = {"torch":torch, "omni.ext._impl": SimpleNamespace(stat_cache=SimpleNamespace(reset_stat_cache=cached.clear))}
        with patch.dict(sys.modules, modules):
            collect_environment([("environment", ref)])
        self.assertIsNone(ref())

    def test_expired_usd_stage_wrapper_is_not_a_live_native_stage(self):
        class Stage:
            valid = True
            def __bool__(self):
                return self.valid
        stage = Stage()
        ref = weakref.ref(stage)
        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:None, empty_cache=lambda:None,
            memory_allocated=lambda:0, memory_reserved=lambda:0))
        with patch.dict(sys.modules, {"torch":torch}), patch("utils.service_session.clear_import_cache"):
            with self.assertRaisesRegex(RuntimeError, "stage"):
                collect_environment([("stage", ref)])
            stage.valid = False
            collect_environment([("stage", ref)])
            # Never apply USD's special validity rule to ordinary falsey owners.
            with self.assertRaisesRegex(RuntimeError, "environment"):
                collect_environment([("environment", ref)])

    def test_shutdown_signal_unwinds_but_does_not_reenter_cleanup(self):
        owner = EnvironmentOwner(SimpleNamespace(close=lambda:self.fail("handler must not close app")))
        with self.assertRaises(SessionShutdown):
            owner.signal(15, None)
        owner.closing = True
        owner.signal(15, None)
        self.assertTrue(owner.stopping)

    def test_progress_transfer_has_no_environment_or_mutable_alias(self):
        env = SimpleNamespace(video_writers={}, save_dir="result", success_nums=1, fail_nums=2,
            total_score=1.5, eval_result={"details":{0:{"layout_id":7}}}, abandoned_seeds=set())
        progress = completed_progress(env)
        env.eval_result["details"][0]["layout_id"] = 99
        self.assertEqual(progress["details"]["0"]["layout_id"], 7)
        self.assertEqual(progress["fail_nums"], 2)

    def test_mailbox_ack_waits_for_cleanup_and_identical_jobs_rebuild(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/"eval_result").mkdir()
            mailbox = root/"mailbox"
            mailbox.mkdir()
            request = self.request()
            (mailbox/"request.json").write_text(json.dumps(request))
            owner = EnvironmentOwner(SimpleNamespace())
            released, replies = [], []
            def run_job(*, resident, owner):
                self.assertIsNone(owner.env)
                env = SimpleNamespace(sim=object(), num_envs=1, success_nums=0, fail_nums=1)
                owner.adopt(env)
                return env
            def detach():
                self.assertIsNotNone(owner.env)
                owner.env = None
                return ["retired"]
            def collect(refs):
                self.assertEqual(refs, ["retired"])
                released.append(True)
            def reply(path, value):
                self.assertIsNone(owner.env)
                self.assertEqual(len(released), len(replies)+1)
                self.assertTrue(value["environment_released"])
                self.assertEqual(value["environment_builds"], 1)
                self.assertEqual(value["code"], 0)
                replies.append(value)
                if len(replies) == 2:
                    raise SessionShutdown()
                request.update(sequence=2, job_id="e"*32)
                (mailbox/"request.json").write_text(json.dumps(request))
            previous = os.getcwd()
            try:
                os.chdir(root)
                with patch.dict(os.environ, {"SIM_SERVICE_SESSION_ID":"a"*32}), \
                        patch.object(owner,"detach",detach), patch.object(owner,"collect",collect), \
                        patch("utils.service_session.atomic_json",reply):
                    with self.assertRaises(SessionShutdown):
                        _serve_jobs(SimpleNamespace(service_session=str(mailbox)), run_job, object(), WallProfile(), owner)
            finally:
                os.chdir(previous)
            self.assertEqual([r["mode"] for r in replies], ["cold","rebuild"])
            self.assertEqual([r["environment_generation"] for r in replies], [1,2])

    def test_service_close_retains_partial_video_instead_of_abort(self):
        events = []
        close = load_method("src/eval_client/eval_env.py", "EvalEnv", "close",
            {"os":os, "super":lambda:SimpleNamespace(close=lambda:events.append("env"))})
        writer = SimpleNamespace(close=lambda **kwargs:events.append("video-close"))
        env = SimpleNamespace(video_writers={0:{"head":writer}},
            obs_manager=SimpleNamespace(reset=lambda:events.append("obs")),
            _abort_video_writers=lambda:self.fail("partial evidence cannot be deleted"))
        with patch.dict(os.environ, {"SIM_SERVICE_SESSION_ID":"a"*32}):
            close(env)
        self.assertEqual(events, ["video-close","obs","env"])
        self.assertEqual(env.video_writers, {})

    def test_failed_job_cleans_up_and_never_acknowledges_success(self):
        for failure in ("job", "cleanup"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "eval_result").mkdir()
                mailbox = root / "mailbox"
                mailbox.mkdir()
                (mailbox / "request.json").write_text(json.dumps(self.request()))
                owner = EnvironmentOwner(object())
                replies, releases = [], []
                def run_job(**kwargs):
                    env = SimpleNamespace(sim=object(), num_envs=1, success_nums=0, fail_nums=1)
                    owner.adopt(env)
                    if failure == "job":
                        raise RuntimeError("injected job error")
                    return env
                def detach():
                    owner.env = None
                    return ["refs"]
                def collect(refs):
                    releases.append(refs)
                    if failure == "cleanup":
                        raise RuntimeError("injected cleanup error")
                previous = os.getcwd()
                try:
                    os.chdir(root)
                    with patch.dict(os.environ, {"SIM_SERVICE_SESSION_ID":"a"*32}), \
                            patch.object(owner,"detach",detach), patch.object(owner,"collect",collect), \
                            patch("utils.service_session.atomic_json",lambda path,value:replies.append(value)):
                        _serve_jobs(SimpleNamespace(service_session=str(mailbox)), run_job, object(), WallProfile(), owner)
                finally:
                    os.chdir(previous)
                self.assertEqual(len(replies), 1)
                self.assertEqual(replies[0]["code"], 1)
                self.assertIn(f"injected {failure} error", replies[0]["error"])
                self.assertEqual(releases, [["refs"]])

    def test_outer_shutdown_also_closes_app_after_partial_construction_failure(self):
        events = []
        # Simulates context registration before create_eval_env can return and
        # transfer ownership. The process must retire instead of serving again.
        simulation = SimpleNamespace(_app_control_on_stop_handle=SimpleNamespace(unsubscribe=lambda:events.append("unsubscribe")),
            _on_post_physics_ready_callback=SimpleNamespace(reset=lambda:events.append("ready-reset")))
        modules = {"isaaclab.sim":SimpleNamespace(SimulationContext=SimpleNamespace(instance=lambda:simulation))}
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, modules), \
                patch("utils.service_session._serve_jobs", side_effect=RuntimeError("construction")), \
                patch("utils.service_session.collect_environment", return_value={}), \
                patch("utils.service_session.signal.signal", return_value=None):
            app = SimpleNamespace(close=lambda:events.append("application-close"))
            with self.assertRaisesRegex(RuntimeError, "construction"):
                serve(SimpleNamespace(service_session=temporary), None, app, WallProfile())
            receipt = json.loads((Path(temporary)/"shutdown.json").read_text())
        self.assertFalse(receipt["environment_owned"])
        self.assertIsNone(receipt["cleanup_error"])
        self.assertEqual(events, ["unsubscribe", "ready-reset", "application-close"])

    def test_configure_evaluation_resets_all_job_state_without_replacing_sim(self):
        namespace = {"deepcopy": deepcopy, "os": os, "datetime": datetime, "BENCHMARK": "RoboDojo",
                     "ObsManager": DummyManager, "SeedManager": DummyManager, "WsModelClient": DummyManager,
                     "get_robot_action_dim_info": lambda **kw: {}, "profiled": lambda name: lambda fn: fn,
                     "_patch_websockets_proxy_compat": lambda: None}
        configure = load_method("src/eval_client/eval_env.py", "EvalEnv", "configure_evaluation", namespace)
        env = SimpleNamespace(num_envs=1, dt=.004, sim=object(), video_writers={},
                              scene_manager=SimpleNamespace(layout_manager=SimpleNamespace(replay=False)),
                              _sweep_stream_dir=lambda: None)
        config = SimpleNamespace(sim=SimpleNamespace(scene=SimpleNamespace(num_envs=1), seed=[0]),
                                 eval_cfg={"task_name": "stack_bowls", "policy_name": "Pi_05",
                                           "config_name": "arx_x5", "eval_num": 1, "seed": 0},
                                 deploy_cfg={"port": 9999})
        with patch.dict(os.environ, {"ROBODOJO_RUN_ID": "service-first"}):
            configure(env, config)
        client, seed_manager, obs_manager, simulator = env.model_client, env.seed_manager, env.obs_manager, env.sim
        env.success_nums, env.fail_nums, env.total_score, env.unstable_nums = 3, 7, 9, 5
        env.eval_result["details"][0] = {"layout_id": 8}
        env.abandoned_seeds.add(9)
        env.current_env_seed_map[0] = 8
        config.eval_cfg["seed"] = 1
        with patch.dict(os.environ, {"ROBODOJO_RUN_ID": "service-next", "ROBODOJO_OUTPUT_ROOT": "eval_result/new"}):
            configure(env, config)
        self.assertIs(env.sim, simulator)
        self.assertTrue(client.closed)
        self.assertIsNot(env.model_client, client)
        self.assertIsNot(env.seed_manager, seed_manager)
        self.assertIsNot(env.obs_manager, obs_manager)
        self.assertEqual((env.success_nums, env.fail_nums, env.total_score, env.unstable_nums), (0, 0, 0, 0))
        self.assertEqual(env.eval_result["details"], {})
        self.assertEqual(env.abandoned_seeds, set())
        self.assertEqual(env.current_env_seed_map, {})
        self.assertEqual(env.eval_seed, 1)
        self.assertTrue(env.save_dir.startswith("eval_result/new/"))
        self.assertTrue(env.save_dir.endswith("service-next"))
        self.assertEqual(env.seed_manager.completed, {"completed_layout_ids": [], "abandoned_layout_ids": []})

    def test_live_video_writer_prevents_reconfiguration(self):
        configure = load_method("src/eval_client/eval_env.py", "EvalEnv", "configure_evaluation", {})
        with self.assertRaises(RuntimeError):
            configure(SimpleNamespace(video_writers={0: object()}), None)

    def test_camera_soft_reset_does_not_duplicate_render_products(self):
        reset = load_method("env/camera_manager/capture/tiled_capture_manager.py", "TiledCaptureManager", "reset", {})
        capture = SimpleNamespace(tiled_cameras=[object()], num_cams=1,
                                  cameras=[[SimpleNamespace(prim_path="/camera")]], camera_prim_paths=[["/camera"]],
                                  init_cameras=lambda: self.fail("must reuse existing camera"))
        reset(capture)
        capture.cameras[0][0].prim_path = "/changed"
        with self.assertRaises(RuntimeError):
            reset(capture)

    def test_renderer_history_cleared_after_layout_before_native_warmup(self):
        events = []
        context = SimpleNamespace(reset_renderer_accumulation=lambda: events.append("history-reset"))
        usd = SimpleNamespace(get_context=lambda: context)
        omni = SimpleNamespace(usd=usd)
        setup = load_method("src/eval_client/eval_env.py", "EvalEnv", "setup_scene", {"os": os})
        env = SimpleNamespace(num_envs=1, unstable_nums=0, episode_nums=1, physx_monitor_enabled=False,
            scene_manager=SimpleNamespace(apply_saved_poses=lambda **kw: events.append("layout"),
                layout_manager=SimpleNamespace(check_layout_stability=lambda env: (True, []))),
            _align_layout_success=lambda: None, render=lambda: events.append("render"),
            sim_step=lambda: events.append("physics"), obs_manager=SimpleNamespace(get_obs=lambda: events.append("obs")))
        with patch.dict(os.environ, {"SIM_SERVICE_SESSION_ID": "a" * 32}), patch.dict(sys.modules, {"omni": omni, "omni.usd": usd}):
            setup(env)
        self.assertEqual(events[:3], ["layout", "history-reset", "render"])
        self.assertEqual(events.count("history-reset"), 1)
        self.assertEqual(events.count("render"), 50)
        self.assertEqual(events.count("physics"), 200)
        self.assertEqual(events.count("obs"), 40)
        self.assertTrue(env._renderer_history_reset)


if __name__ == "__main__":
    unittest.main()
