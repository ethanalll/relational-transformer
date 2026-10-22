"""Explicit target identities, independent of PLuREL and table-name conventions."""

from dataclasses import dataclass
from pathlib import Path
import json


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
