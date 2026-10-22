"""Explicit target identities, independent of PLuREL and table-name conventions."""

from dataclasses import dataclass
from pathlib import Path
import json
import random


@dataclass(frozen=True)
class TaskSpec:
    db_name: str
    table_name: str
    target_column: str
    task_type: str
    columns_to_drop: tuple[str, ...] = ()

    def key(self):
        return (self.db_name, self.table_name, self.target_column)


def load_tasks(pre_dir):
    tasks = []
    for path in sorted(Path(pre_dir).glob("*/export.json")):
        record = json.loads(path.read_text())
        for spec in record["tasks"]:
            spec["columns_to_drop"] = tuple(spec.get("columns_to_drop", ()))
            tasks.append(TaskSpec(db_name=record["name"], **spec))
    if not tasks:
        raise ValueError(f"No eligible tasks under {pre_dir}")
    if len({t.key() for t in tasks}) != len(tasks):
        raise ValueError("Duplicate prediction task")
    return tasks


def select_validation_tasks(tasks, tasks_per_db=None, seed=0):
    """Fixed per-database sample, including both target types when possible.

    The local RNG is independent of training and input task ordering. Leaving
    tasks_per_db unset retains all tasks for broader standalone evaluation.
    """
    tasks = sorted(tasks, key=lambda task: task.key())
    if tasks_per_db is None:
        return tasks
    if type(tasks_per_db) is not int or tasks_per_db < 1:
        raise ValueError("eval_tasks_per_db must be a positive integer or null")
    grouped = {}
    for task in tasks:
        grouped.setdefault(task.db_name, {}).setdefault(task.task_type, []).append(task)
    selected = []
    for db_name, kinds in grouped.items():
        rng = random.Random(f"validation:{seed}:{db_name}")
        for candidates in kinds.values():
            rng.shuffle(candidates)
        # With a cap >=2, include a Boolean and a numerical target if available.
        kind_order = sorted(kinds)
        rng.shuffle(kind_order)
        chosen = [kinds[kind].pop() for kind in kind_order[:tasks_per_db]]
        remaining = [task for candidates in kinds.values() for task in candidates]
        rng.shuffle(remaining)
        chosen.extend(remaining[:max(0, tasks_per_db - len(chosen))])
        selected.extend(chosen)
    return sorted(selected, key=lambda task: task.key())
