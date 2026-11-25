import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from relbench.base import Table

from rt.experiment import dataset, score
from rt.experiment_data import MINILM, ROOT, forecast_policy, relbench_tasks
from rt.model import RelationalTransformer


DB = "fixture"
TASK = "user-ltv"
CLASSIFICATION = "user-churn"
CONFIG = {"batch": 2, "context": 64, "width": 64}
EPOCH = pd.Timestamp("2020-01-01")
DAY = 24 * 60 * 60


def unix(day):
    return EPOCH.value // 1_000_000_000 + day * DAY


def dates(days):
    # The Rust cache must use the same timestamp units for nodes and both edge directions.
    return pd.to_datetime([unix(day) for day in days], unit="s").as_unit("us")


def save(path, data, *, pk=None, fk=None, dated=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    Table(
        df=pd.DataFrame(data),
        pkey_col=pk,
        fkey_col_to_pkey_table=fk or {},
        time_col="date" if dated else None,
    ).save(path)


def build_fixture(root, native, *, val_labels, test_labels, future_value):
    raw = root / "raw"
    dated = raw / "db"
    save(dated / "users.parquet", {"id": [0, 1], "amount": [1.0, 3.0]}, pk="id")
    save(
        dated / "items.parquet",
        {"id": [0, 1], "date": dates([3, 40]), "cost": [3.0, future_value]},
        pk="id",
        dated=True,
    )
    save(
        dated / "review.parquet",
        {
            "id": [0, 1, 2, 3],
            "user": [0, 1, 0, 0],
            "item": [0, 0, 1, 0],
            "date": dates([10, 11, 15, 43]),
            "rating": [2.0, 4.0, 5.0, future_value],
        },
        pk="id",
        fk={"user": "users", "item": "items"},
        dated=True,
    )
    for split, users, timestamps, labels in (
        ("train", [0, 1, 0], [3, 13, 18], [2.0, 6.0, 9.0]),
        ("val", [0, 1, 0], [20, 20, 40], val_labels),
        ("test", [0, 1, 0], [35, 35, 45], test_labels),
    ):
        save(
            raw / "tasks" / TASK / f"{split}.parquet",
            {"user": users, "date": dates(timestamps), "ltv": labels},
            fk={"user": "users"},
            dated=True,
        )
        save(
            raw / "tasks" / CLASSIFICATION / f"{split}.parquet",
            {
                "user": users,
                "date": dates(timestamps),
                "churn": [value > 4.0 for value in labels],
            },
            fk={"user": "users"},
            dated=True,
        )
    cache = root / "pre" / DB
    subprocess.run(
        [
            str(native),
            "pre",
            DB,
            "--input-dir",
            str(raw),
            "--output-dir",
            str(cache),
            "--stats-cutoff",
            str(unix(20)),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    # Embedding values are fixed for this test. The native preprocessor, sampler,
    # normalization, model and scorer all run through their real entry points.
    texts = json.loads((cache / "text.json").read_text())
    (cache / f"text_emb_{MINILM}.bin").write_bytes(bytes(len(texts) * 384 * 2))
    return cache.parent


class RelBenchForecastTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temp.cleanup)
        native = Path(
            os.environ.get(
                "RT_TEST_NATIVE_BIN", ROOT / "rustler/target/release/rustler"
            )
        )
        if not native.is_file():
            raise FileNotFoundError("Build native code first or set RT_TEST_NATIVE_BIN")
        splits = {"val": [False, 7 * DAY, unix(20)], "test": [True, 7 * DAY, unix(35)]}
        cls.policies = {DB: {TASK: splits, CLASSIFICATION: splits}}
        cls.clean = build_fixture(
            Path(temp.name) / "clean",
            native,
            val_labels=[3.0, 5.0, 11.0],
            test_labels=[1.0, 8.0, 13.0],
            future_value=10.0,
        )
        cls.changed_test = build_fixture(
            Path(temp.name) / "test-changed",
            native,
            val_labels=[3.0, 5.0, 11.0],
            test_labels=[130.0, 10.0, 80.0],
            future_value=1000.0,
        )
        cls.changed_val = build_fixture(
            Path(temp.name) / "val-changed",
            native,
            val_labels=[300.0, 50.0, 110.0],
            test_labels=[1.0, 8.0, 13.0],
            future_value=10.0,
        )
        cls.tables = json.loads((cls.clean / DB / "table_info.json").read_text())

    def rows(self, table):
        info = self.tables[table]
        return range(
            info["node_idx_offset"], info["node_idx_offset"] + info["num_nodes"]
        )

    def batches(self, cache, split, *, forecast=True):
        return dataset(
            cache,
            [[DB, TASK, "ltv", [], "reg"]],
            split,
            CONFIG,
            shuffle_seed=0,
            forecast_policies=self.policies if forecast else None,
        )

    def contexts(self, split, *, forecast=True):
        data = self.batches(self.clean, split, forecast=forecast)
        for index in range(len(data)):
            batch = data[index]
            for row in range(batch["true_batch_size"]):
                seed = batch["node_idxs"][row, batch["is_targets"][row]].item()
                context = set(
                    batch["node_idxs"][row, ~batch["is_padding"][row]].tolist()
                )
                yield seed, context - {seed}

    def test_negative_control_reproduces_peer_labels(self):
        peer_test_rows = set(self.rows(f"{TASK}:Test"))
        self.assertTrue(
            any(
                peer_test_rows & peers
                for _, peers in self.contexts("test", forecast=False)
            )
        )

    def test_split_cutoffs_and_complete_label_windows(self):
        train = list(self.rows(f"{TASK}:Train"))
        validation = list(self.rows(f"{TASK}:Val"))
        test = list(self.rows(f"{TASK}:Test"))
        items = list(self.rows("items:Db"))
        review = list(self.rows("review:Db"))
        for split, known, withheld in (
            (
                "val",
                {*train[:2], items[0]},
                {train[2], *validation, *test, items[1], review[3]},
            ),
            (
                "test",
                {*train, *validation[:2], items[0]},
                {validation[2], *test, items[1], review[3]},
            ),
        ):
            with self.subTest(split=split):
                contexts = list(self.contexts(split))
                self.assertEqual(len(contexts), 3)
                for _, peers in contexts:
                    self.assertTrue(known <= peers)
                    self.assertFalse(withheld & peers)

    def test_focal_and_peer_targets_cannot_change_predictions(self):
        torch.manual_seed(0)
        model = RelationalTransformer(
            num_blocks=1,
            d_model=16,
            d_text=384,
            num_heads=2,
            d_ff=32,
        ).eval()
        predictions = []

        def record(_module, args, result):
            batch = args[0]
            mask = batch["is_targets"]
            self.assertTrue(batch["masks"][mask].all())
            self.assertEqual(batch["number_values"][mask].abs().max().item(), 0)
            predictions.append(result[1]["number"][mask].clone())

        with model.register_forward_hook(record):
            for split, changed in (
                ("val", self.changed_val),
                ("test", self.changed_test),
            ):
                with self.subTest(split=split):
                    reference = score(
                        model,
                        self.batches(self.clean, split),
                        "reg",
                        None,
                        forecast=True,
                    )
                    expected = torch.cat(predictions)
                    predictions.clear()
                    rerun = score(
                        model, self.batches(changed, split), "reg", None, forecast=True
                    )
                    self.assertEqual(reference["examples"], rerun["examples"])
                    self.assertNotEqual(reference["loss"], rerun["loss"])
                    torch.testing.assert_close(
                        torch.cat(predictions), expected, rtol=0, atol=0
                    )
                    predictions.clear()

    def test_classification_labels_only_change_the_score(self):
        torch.manual_seed(0)
        model = RelationalTransformer(
            num_blocks=1,
            d_model=16,
            d_text=384,
            num_heads=2,
            d_ff=32,
        ).eval()
        predictions = []

        def record(_module, args, result):
            batch = args[0]
            mask = batch["is_targets"]
            self.assertEqual(batch["boolean_values"][mask].abs().max().item(), 0)
            predictions.append(result[1]["boolean"][mask].clone())

        losses = []
        results = []
        with model.register_forward_hook(record):
            for cache in (self.clean, self.changed_test):
                data = dataset(
                    cache,
                    [[DB, CLASSIFICATION, "churn", [], "clf"]],
                    "test",
                    CONFIG,
                    shuffle_seed=0,
                    forecast_policies=self.policies,
                )
                losses.append(score(model, data, "clf", None, forecast=True)["loss"])
                results.append(torch.cat(predictions))
                predictions.clear()
        self.assertNotEqual(*losses)
        torch.testing.assert_close(*results, rtol=0, atol=0)

    def test_every_configured_task_has_official_horizon_and_cutoffs(self):
        tasks = relbench_tasks("all")
        self.assertEqual(len(tasks), 18)
        for db, table, *_ in tasks:
            with self.subTest(db=db, task=table):
                validation = forecast_policy(db, table, "val")
                test = forecast_policy(db, table, "test")
                self.assertFalse(validation[0])
                self.assertTrue(test[0])
                self.assertGreater(validation[1], 0)
                self.assertEqual(validation[1], test[1])
                self.assertGreaterEqual(test[2], validation[2] + validation[1])
