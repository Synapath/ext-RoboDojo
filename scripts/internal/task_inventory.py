#!/usr/bin/env python3
"""List runnable RoboDojo tasks without importing Isaac-dependent task modules."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
BENCHMARK = "RoboDojo"
TASK_DIR = ROOT_DIR / "task" / BENCHMARK / "tasks"
CONFIG_DIR = ROOT_DIR / "task" / BENCHMARK / "config"
POLICY_DIR = ROOT_DIR / "XPolicyLab" / "policy"
ENV_CONFIG_DIR = ROOT_DIR / "env_cfg"
SERVICE_SCHEMA = "robodojo-runtime-inventory-v1"

sys.path.insert(0, str(ROOT_DIR))
from task.RoboDojo import task_registry  # noqa: E402


# Canonical capability groups from https://robodojo-benchmark.com/doc/sim-tasks/.
# Generalization tasks also have runnable ``*_random`` configurations; those
# inherit the dimension of their base task in _task_records().
DIMENSION_TASKS: dict[str, tuple[str, ...]] = {
    "generalization": (
        "stack_bowls",
        "push_T",
        "pack_objects_into_box",
        "fold_clothes",
        "hang_mugs",
        "sweep_blocks",
        "pour_liquid_into_cup",
        "make_toast",
        "arrange_largest_number",
        "sort_nesting_dolls_by_size",
        "store_laptop_and_headphones",
        "stack_blocks",
    ),
    "memory": (
        "cover_blocks",
        "match_and_pick_from_conveyor",
        "swap_blocks",
        "swap_T",
        "press_by_number",
        "imitate_sorting_sequence",
    ),
    "precision": (
        "fasten_screws",
        "plug_in_charger",
        "insert_tubes",
        "pour_balls_into_vase",
        "play_Xylophone",
        "deposit_coin",
        "insert_key",
        "build_tower",
    ),
    "long-horizon": (
        "fill_pen_holder",
        "classify_objects",
        "put_bottles_into_dustbin",
        "play_tic_tac_toe",
        "fill_egg_holder",
        "organize_table",
        "make_kong",
        "play_stacking_toy",
    ),
    "open": (
        "align_blocks",
        "general_pickup",
        "solve_equation",
        "stack_blocks_by_language",
        "classify_objects_by_language",
        "pick_from_conveyor_by_image",
        "store_tools_in_toolbox",
        "pour_by_language",
    ),
}

DIMENSION_ALIASES = {
    "gen": "generalization",
    "generalisation": "generalization",
    "mem": "memory",
    "long": "long-horizon",
    "long_horizon": "long-horizon",
    "longhorizon": "long-horizon",
    "open-ended": "open",
    "open_ended": "open",
    "openended": "open",
}


def _dimension_for_task(name: str) -> str | None:
    base_name = name.removesuffix("_random")
    for dimension, tasks in DIMENSION_TASKS.items():
        if base_name in tasks:
            return dimension
    return None


def _parse_dimensions(values: list[str]) -> list[str]:
    dimensions = []
    select_all = False
    for value in values:
        items = value.split(",")
        if any(not item.strip() for item in items):
            raise ValueError("dimension names must not be empty")
        for item in items:
            name = item.strip().lower()
            name = DIMENSION_ALIASES.get(name, name)
            if name == "all":
                select_all = True
                continue
            if name not in DIMENSION_TASKS:
                choices = ", ".join(DIMENSION_TASKS)
                raise ValueError(f"unknown dimension '{item.strip()}' (choose from: {choices}, all)")
            if name not in dimensions:
                dimensions.append(name)
    if select_all:
        return list(DIMENSION_TASKS)
    return dimensions


def _normalized_dimension_names(dimensions: list[str]) -> list[str]:
    if not dimensions or set(dimensions) == set(DIMENSION_TASKS):
        return ["all"]
    return dimensions


def _module_classes(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {node.name for node in tree.body if isinstance(node, ast.ClassDef)}


def _task_records() -> list[dict[str, Any]]:
    records = []
    for path in sorted(TASK_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        name = path.stem
        classes = _module_classes(path)
        config_path = task_registry.task_config_path(CONFIG_DIR, name)
        record = {
            "name": name,
            "dimension": _dimension_for_task(name),
            "variant": "random" if name.endswith("_random") else "standard",
            "module": str(path.relative_to(ROOT_DIR)),
            "class_name": name,
            "class_exists": name in classes,
            "config": str(config_path.relative_to(ROOT_DIR)) if config_path.exists() else None,
            "config_exists": config_path.exists(),
        }
        record["runnable"] = bool(record["class_exists"] and record["config_exists"])
        records.append(record)
    return records


def _inventory_counts(tasks: list[dict[str, Any]], config_only: list[str]) -> dict[str, int]:
    return {
        "tasks": len(tasks),
        "runnable": sum(1 for task in tasks if task["runnable"]),
        "missing_config": sum(1 for task in tasks if not task["config_exists"]),
        "missing_class": sum(1 for task in tasks if not task["class_exists"]),
        "config_only": len(config_only),
    }


def build_inventory() -> dict[str, Any]:
    tasks = _task_records()
    task_names = {task["name"] for task in tasks}
    config_names = {path.stem for path in CONFIG_DIR.glob("*.yml") if not path.name.startswith("_")}
    config_only = sorted(config_names - task_names)
    inventory = {
        "benchmark": BENCHMARK,
        "root": str(ROOT_DIR),
        "task_dir": str(TASK_DIR.relative_to(ROOT_DIR)),
        "config_dir": str(CONFIG_DIR.relative_to(ROOT_DIR)),
        "counts": _inventory_counts(tasks, config_only),
        "tasks": tasks,
        "config_only": config_only,
    }
    return inventory


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be an object: {path}")
    return value


def _function_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def build_service_inventory(env_configs: list[str]) -> dict[str, Any]:
    """Pure filesystem inventory for sim-service startup; imports no simulator."""
    task_index = _load_yaml(CONFIG_DIR / "_task.yml")
    common = task_index.get("common")
    overrides = task_index.get("tasks")
    if not isinstance(common, dict) or not isinstance(overrides, dict):
        raise ValueError("RoboDojo task index is invalid")
    tasks = {}
    for record in _task_records():
        if not record["runnable"]:
            raise ValueError(f"task class/config pair is incomplete: {record['name']}")
        task_override = overrides.get(record["name"], {})
        if not isinstance(task_override, dict):
            raise ValueError(f"task override is invalid: {record['name']}")
        native_eval_num = task_override.get("eval_nums", common.get("eval_nums"))
        if type(native_eval_num) is not int or native_eval_num <= 0:
            raise ValueError(f"task eval_nums is invalid: {record['name']}")
        tasks[record["name"]] = {
            "native_eval_num": native_eval_num,
            "max_num_envs": None,
        }

    adapters = {}
    if not POLICY_DIR.is_dir() or POLICY_DIR.is_symlink():
        raise ValueError("XPolicyLab policy directory is unavailable")
    for directory in sorted(POLICY_DIR.iterdir()):
        if directory.name.startswith("."):
            continue
        if not directory.is_dir() and not directory.is_symlink():
            continue
        deploy_yml = directory / "deploy.yml"
        deploy_py = directory / "deploy.py"
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"policy adapter directory is invalid: {directory.name}")
        if not deploy_yml.is_file() or deploy_yml.is_symlink():
            continue
        if not deploy_py.is_file() or deploy_py.is_symlink():
            raise ValueError(f"policy adapter entrypoint is missing: {directory.name}")
        deploy = _load_yaml(deploy_yml)
        if deploy.get("policy_name") != directory.name:
            raise ValueError(f"policy adapter name mismatch: {directory.name}")
        eval_batch = deploy.get("eval_batch", False)
        if type(eval_batch) is not bool:
            raise ValueError(f"policy adapter eval_batch is invalid: {directory.name}")
        functions = _function_names(deploy_py)
        entrypoint = "batch" if eval_batch else "single"
        required = "eval_one_episode_batch" if eval_batch else "eval_one_episode"
        if required not in functions:
            raise ValueError(f"policy adapter entrypoint is missing: {directory.name}")
        adapters[directory.name] = {
            "eval_batch": eval_batch,
            "execution_horizon_required": directory.name == "Pi_05",
            "entrypoint": entrypoint,
        }
    if not adapters:
        raise ValueError("no policy adapters were discovered")

    environments = {}
    for name in env_configs:
        if not name or Path(name).name != name:
            raise ValueError(f"invalid env config name: {name}")
        path = ENV_CONFIG_DIR / f"{name}.yml"
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"env config is missing: {name}")
        value = _load_yaml(path)
        config = value.get("config")
        sim_name = config.get("sim") if isinstance(config, dict) else None
        if not isinstance(sim_name, str) or not sim_name:
            raise ValueError(f"env config sim declaration is invalid: {name}")
        sim_path = ENV_CONFIG_DIR / "sim" / f"{sim_name}.yml"
        sim = _load_yaml(sim_path)
        scene = sim.get("scene")
        num_envs = scene.get("num_envs") if isinstance(scene, dict) else None
        if type(num_envs) is not int or num_envs <= 0:
            raise ValueError(f"sim num_envs is invalid: {sim_name}")
        random_task_num_envs = scene.get("clutter_env_limit", 5)
        if type(random_task_num_envs) is not int or random_task_num_envs <= 0:
            raise ValueError(f"sim clutter_env_limit is invalid: {sim_name}")
        environments[name] = {
            "num_envs": num_envs,
            "random_task_num_envs": random_task_num_envs,
            "sim_config": sim_name,
        }
    return {
        "schema_version": SERVICE_SCHEMA,
        "tasks": tasks,
        "policy_adapters": adapters,
        "env_configs": environments,
    }


def _print_plain(inventory: dict[str, Any], only_runnable: bool) -> None:
    for task in inventory["tasks"]:
        if only_runnable and not task["runnable"]:
            continue
        print(task["name"])


def _print_markdown(inventory: dict[str, Any]) -> None:
    print("| Task | Dimension | Variant | Runnable | Config | Issue |")
    print("| --- | --- | --- | --- | --- | --- |")
    for task in inventory["tasks"]:
        issues = []
        if not task["class_exists"]:
            issues.append("missing exported class")
        if not task["config_exists"]:
            issues.append("missing config")
        print(
            f"| `{task['name']}` | {task['dimension'] or '-'} | {task['variant']} | "
            f"{'yes' if task['runnable'] else 'no'} | "
            f"`{task['config'] or '-'}` | {', '.join(issues) or '-'} |"
        )
    if inventory["config_only"]:
        print("\nConfig-only entries:")
        for name in inventory["config_only"]:
            print(f"- `{name}`")


def _print_dimensions(inventory: dict[str, Any], output_format: str) -> None:
    runnable = [task for task in inventory["tasks"] if task["runnable"]]
    rows = []
    for dimension, canonical_tasks in DIMENSION_TASKS.items():
        runtime_tasks = [task["name"] for task in runnable if task["dimension"] == dimension]
        rows.append(
            {
                "dimension": dimension,
                "canonical_count": len(canonical_tasks),
                "runnable_count": len(runtime_tasks),
                "tasks": list(canonical_tasks),
                "runnable_tasks": runtime_tasks,
            }
        )
    if output_format == "json":
        print(json.dumps(rows, indent=2))
    elif output_format == "markdown":
        print("| Dimension | Canonical | Runnable configs | Tasks |")
        print("| --- | ---: | ---: | --- |")
        for row in rows:
            print(
                f"| {row['dimension']} | {row['canonical_count']} | {row['runnable_count']} | "
                f"{', '.join(f'`{task}`' for task in row['tasks'])} |"
            )
    else:
        for row in rows:
            print(
                f"{row['dimension']} ({row['canonical_count']} canonical, "
                f"{row['runnable_count']} runnable configs)"
            )
            for task in row["runnable_tasks"]:
                print(f"  {task}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=("plain", "json", "markdown"),
        default="plain",
        help="Output format. Use json for agents and plain for shell loops.",
    )
    parser.add_argument(
        "--only-runnable",
        action="store_true",
        help="Only print runnable task names in plain output.",
    )
    parser.add_argument(
        "--dimension",
        action="append",
        default=[],
        metavar="NAME[,NAME...]",
        help="Filter tasks by capability dimension; may be repeated (or use all).",
    )
    parser.add_argument(
        "--list-dimensions",
        action="store_true",
        help="List the official capability dimensions and their tasks.",
    )
    parser.add_argument(
        "--resolve-dimensions",
        action="store_true",
        help="Validate dimensions and print their canonical, de-duplicated names.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if any task is missing its config or exported class.",
    )
    parser.add_argument(
        "--service-inventory",
        action="store_true",
        help="Emit the strict CPU-only sim-service runtime inventory JSON.",
    )
    parser.add_argument(
        "--env-config",
        action="append",
        default=[],
        help="Deployment env config to validate in --service-inventory mode.",
    )
    args = parser.parse_args()

    try:
        selected_dimensions = _parse_dimensions(args.dimension)
    except ValueError as exc:
        parser.error(str(exc))

    if args.service_inventory:
        if not args.env_config:
            parser.error("--service-inventory requires at least one --env-config")
        try:
            value = build_service_inventory(args.env_config)
        except ValueError as exc:
            parser.error(str(exc))
        print(json.dumps(value, sort_keys=True))
        return 0

    if args.env_config:
        parser.error("--env-config is only valid with --service-inventory")

    if args.resolve_dimensions:
        print(",".join(_normalized_dimension_names(selected_dimensions)))
        return 0

    inventory = build_inventory()
    if args.list_dimensions:
        _print_dimensions(inventory, args.format)
    else:
        if selected_dimensions:
            inventory = dict(inventory)
            inventory["tasks"] = [
                task for task in inventory["tasks"] if task["dimension"] in selected_dimensions
            ]
            inventory["counts"] = _inventory_counts(
                inventory["tasks"], inventory["config_only"]
            )
        if args.format == "json":
            print(json.dumps(inventory, indent=2, sort_keys=True))
        elif args.format == "markdown":
            _print_markdown(inventory)
        else:
            _print_plain(inventory, only_runnable=args.only_runnable)

    if args.check:
        all_tasks = build_inventory()["tasks"]
        broken = [task for task in all_tasks if not task["runnable"] or not task["dimension"]]
        if broken:
            for task in broken:
                issues = []
                if not task["runnable"]:
                    issues.append("not runnable")
                if not task["dimension"]:
                    issues.append("missing capability dimension")
                print(f"[ERROR] Task {task['name']}: {', '.join(issues)}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
