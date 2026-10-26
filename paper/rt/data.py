"""Paper Rust sampler, plus an explicit portable diagnostic backend."""

import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .embed import EMBEDDING_MODEL


def _load_column_index(pre_dir, db_name):
    return json.loads((Path(pre_dir) / db_name / "column_index.json").read_text())


def get_column_index(column_name, table_name, db_name, pre_dir):
    return _load_column_index(pre_dir, db_name)[f"{column_name} of {table_name}"]


class RelationalDataset(Dataset):
    def __init__(
        self,
        tasks,
        pre_dir,
        batch_size=128,
        ctx_len=1024,
        max_bfs_width=128,
        seed=0,
        backend="native",
    ):
        if not tasks or batch_size < 1 or ctx_len < 2 or max_bfs_width < 1:
            raise ValueError("Invalid sampler configuration")
        self.tasks, self.pre_dir = list(tasks), Path(pre_dir).resolve()
        self.batch_size, self.ctx_len, self.width = batch_size, ctx_len, max_bfs_width
        self.seed, self.epoch, self.backend = seed, 0, backend
        self.databases, self.columns, self.embeddings = {}, {}, {}
        tuples, targets, drops = [], [], []
        for task in tasks:
            db = task.db_name
            folder = self.pre_dir / db
            meta = json.loads((folder / "meta.json").read_text())
            if meta["backend"] != backend:
                raise ValueError("Sampler/cache backend mismatch")
            if db not in self.columns:
                self.columns[db] = _load_column_index(self.pre_dir, db)
                if backend == "portable":
                    self.databases[db] = torch.load(folder / "portable.pt", weights_only=True)[
                        "nodes"
                    ]
                    self.embeddings[db] = torch.load(
                        folder / "embeddings.pt", weights_only=True
                    ).float()
            info = json.loads((folder / "table_info.json").read_text())[f"{task.table_name}:Db"]
            tuples.append((db, info["node_idx_offset"], info["num_nodes"]))
            targets.append(self.columns[db][f"{task.target_column} of {task.table_name}"])
            drops.append(
                [self.columns[db][f"{c} of {task.table_name}"] for c in task.columns_to_drop]
            )
        if backend == "native":
            # Register NumPy's bfloat16 dtype before Rust returns half::bf16 arrays.
            import ml_dtypes  # noqa: F401

            from rustler import Sampler

            self.sampler = Sampler(
                dataset_tuples=tuples,
                batch_size=batch_size,
                rank=0,
                world_size=1,
                ctx_len=ctx_len,
                max_bfs_width=max_bfs_width,
                embedding_model=EMBEDDING_MODEL,
                d_text=384,
                seed=seed,
                target_columns=targets,
                columns_to_drop=drops,
                pre_dir=str(self.pre_dir),
            )
        elif backend == "portable":
            self.items = []
            for task_idx, (db, offset, count) in enumerate(tuples):
                col = targets[task_idx]
                self.items.extend(
                    (task_idx, row)
                    for row in range(offset, offset + count)
                    if any(c["col"] == col for c in self.databases[db][row]["cells"])
                )
            if not self.items:
                raise ValueError("No observed target cells")
            self.targets, self.drops = targets, drops
        else:
            raise ValueError(f"Unknown backend {backend}")
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.epoch = epoch
        if self.backend == "native":
            self.sampler.shuffle_py(epoch)
        else:
            self.order = list(range(len(self.items)))
            random.Random((self.seed << 32) | epoch).shuffle(self.order)

    def __len__(self):
        return (
            self.sampler.len_py()
            if self.backend == "native"
            else math.ceil(len(self.items) / self.batch_size)
        )

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.backend == "native":
            out = dict(self.sampler.batch_py(index))
            count = out.pop("true_batch_size")
            floats = {
                "number_values": 1,
                "datetime_values": 1,
                "boolean_values": 1,
                "text_values": 384,
                "col_name_values": 384,
            }
            for k, v in out.items():
                if k in floats:
                    v = torch.from_numpy(v.view(np.uint16)).view(torch.bfloat16)
                    out[k] = v.reshape(self.batch_size, self.ctx_len, floats[k])[:count]
                else:
                    v = torch.from_numpy(v)
                    shape = (
                        (self.batch_size, self.ctx_len, 5)
                        if k == "f2p_nbr_idxs"
                        else (self.batch_size, self.ctx_len)
                    )
                    out[k] = v.reshape(shape)[:count]
        else:
            indices = self.order[index * self.batch_size : (index + 1) * self.batch_size]
            examples = [self._context(*self.items[i]) for i in indices]
            out = {k: torch.stack([e[k] for e in examples]) for k in examples[0]}
        if not torch.all(out["is_targets"].sum(1) == 1):
            raise ValueError(
                "Each context must contain one target; increase context size or inspect task"
            )
        if any(not torch.isfinite(v).all() for v in out.values() if v.is_floating_point()):
            raise ValueError("Nonfinite sampler output")
        return out

    def _context(self, task_idx, row):
        task = self.tasks[task_idx]
        nodes = self.databases[task.db_name]
        rng = random.Random(((self.seed << 32) | self.epoch) ^ row)
        # Same traversal rules as the paper; Python RNG != Rust RNG (diagnostic only).
        forward, reverse, visited, collected, num_cells = [], {0: [row]}, {}, [], 0
        while forward or any(reverse.values()):
            if forward:
                depth, current = forward.pop()
            else:
                depth = min(d for d, entries in reverse.items() if entries)
                j = rng.randrange(len(reverse[depth]))
                current = reverse[depth].pop(j)
            if current in visited and visited[current] <= depth:
                continue
            num_cells += len(nodes[current]["cells"])
            if num_cells >= self.ctx_len:
                break
            visited[current] = depth
            if current not in collected:
                collected.append(current)
            forward.extend((depth + 1, p) for p in nodes[current]["f2p"])
            timestamp = nodes[row]["timestamp"]
            children = [
                c
                for c in nodes[current]["p2f"]
                if nodes[c]["timestamp"] is None
                or (timestamp is not None and nodes[c]["timestamp"] <= timestamp)
            ]
            if len(children) > self.width:
                children = rng.sample(children, self.width)
            reverse.setdefault(depth + 1, []).extend(children)
        cells = [
            (n, c)
            for n in collected
            for c in nodes[n]["cells"]
            if not (
                c["col"] in self.drops[task_idx]
                and (n == row or nodes[n]["timestamp"] == nodes[row]["timestamp"])
            )
        ]
        cells.sort(key=lambda pair: pair[1]["col"])
        s = self.ctx_len
        out = {
            k: torch.zeros(s, dtype=torch.long)
            for k in ["node_idxs", "col_name_idxs", "table_name_idxs", "sem_types", "timestamps"]
        }
        out["node_idxs"].fill_(-1)
        out["f2p_nbr_idxs"] = torch.full((s, 5), -1, dtype=torch.long)
        out.update(
            {k: torch.zeros(s, dtype=torch.bool) for k in ["masks", "is_targets", "is_task_nodes"]}
        )
        out["is_padding"] = torch.ones(s, dtype=torch.bool)
        for k, d in [
            ("number_values", 1),
            ("boolean_values", 1),
            ("datetime_values", 1),
            ("text_values", 384),
            ("col_name_values", 384),
        ]:
            out[k] = torch.zeros(s, d)
        names = ["number_values", "text_values", "datetime_values", "boolean_values"]
        for i, (node_idx, cell) in enumerate(cells[:s]):
            node = nodes[node_idx]
            out["node_idxs"][i] = node_idx
            out["col_name_idxs"][i] = cell["col"]
            out["table_name_idxs"][i] = node["table"]
            out["sem_types"][i] = cell["kind"]
            out["timestamps"][i] = node["timestamp"] if node["timestamp"] is not None else -(2**31)
            out["f2p_nbr_idxs"][i, : len(node["f2p"])] = torch.tensor(node["f2p"], dtype=torch.long)
            out["col_name_values"][i] = self.embeddings[task.db_name][cell["col"]]
            value = (
                self.embeddings[task.db_name][int(cell["value"])]
                if cell["kind"] == 1
                else cell["value"]
            )
            out[names[cell["kind"]]][i] = value
            target = node_idx == row and cell["col"] == self.targets[task_idx]
            out["masks"][i] = out["is_targets"][i] = target
            out["is_task_nodes"][i] = cell["col"] == self.targets[task_idx]
            out["is_padding"][i] = False
        return out
