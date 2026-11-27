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


def build_fixture(
    root,
    native,
    *,
    val_labels,
    test_labels,
    future_value,
    incomplete_train_label=9.0,
):
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
        ("train", [0, 1, 0], [3, 13, 18], [2.0, 6.0, incomplete_train_label]),
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
            "--task-horizon",
            f"{TASK}={7 * DAY}",
            "--task-horizon",
            f"{CLASSIFICATION}={7 * DAY}",
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
        cls.changed_incomplete_train = build_fixture(
            Path(temp.name) / "incomplete-train-changed",
            native,
            val_labels=[3.0, 5.0, 11.0],
            test_labels=[1.0, 8.0, 13.0],
            future_value=10.0,
            incomplete_train_label=9000.0,
        )
        cls.native = native
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

    def test_incomplete_horizon_train_row_does_not_leak_into_normalization(self):
        clean_batch = self.batches(self.clean, "val")[0]
        changed_batch = self.batches(self.changed_incomplete_train, "val")[0]
        torch.testing.assert_close(
            clean_batch["number_values"],
            changed_batch["number_values"],
            rtol=0,
            atol=0,
        )

    def test_undated_entity_seed_reaches_dated_activity_p2f_children(self):
        # Non-forecast synthetic-style query on undated Entity table (`users`)
        # must still traverse p2f edges to dated child table (`review`).
        data = dataset(
            self.clean,
            [[DB, "users", "amount", [], "reg"]],
            "train",
            CONFIG,
            shuffle_seed=0,
        )
        review_rows = set(self.rows("review:Db"))
        reached = set()
        for index in range(len(data)):
            batch = data[index]
            for row in range(batch["true_batch_size"]):
                reached.update(
                    batch["node_idxs"][row, ~batch["is_padding"][row]].tolist()
                )
        self.assertTrue(review_rows <= reached)

    def test_tiny_context_retains_wide_seed_target_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            wide = {"id": [0, 1]}
            for i in range(8):
                wide[f"f{i}"] = [float(i), float(i + 10)]
            save(raw / "db" / "wide.parquet", wide, pk="id")
            cache = root / "pre" / "wide_db"
            subprocess.run(
                [
                    str(self.native),
                    "pre",
                    "wide_db",
                    "--input-dir",
                    str(raw),
                    "--output-dir",
                    str(cache),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            texts = json.loads((cache / "text.json").read_text())
            (cache / f"text_emb_{MINILM}.bin").write_bytes(bytes(len(texts) * 384 * 2))
            data = dataset(
                cache.parent,
                [["wide_db", "wide", "f7", [], "reg"]],
                "train",
                {"batch": 2, "context": 4, "width": 8},
                shuffle_seed=0,
            )
            batch = data[0]
            self.assertEqual(batch["is_targets"].sum(dim=1).tolist(), [1, 1])
            self.assertTrue(batch["masks"][batch["is_targets"]].all())

    def test_short_span_datetime_normalization_avoids_cancellation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            # Two timestamps 1 hour apart in 2025 (~1.74e18 ns); naive E[X^2]-E[X]^2 loses precision.
            ts = pd.to_datetime(["2025-06-01 00:00:00", "2025-06-01 01:00:00"]).as_unit(
                "ns"
            )
            save(
                raw / "db" / "events.parquet",
                {"id": [0, 1], "date": ts, "val": [1.0, 2.0]},
                pk="id",
                dated=True,
            )
            cache = root / "pre" / "short_dt"
            subprocess.run(
                [
                    str(self.native),
                    "pre",
                    "short_dt",
                    "--input-dir",
                    str(raw),
                    "--output-dir",
                    str(cache),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            texts = json.loads((cache / "text.json").read_text())
            (cache / f"text_emb_{MINILM}.bin").write_bytes(bytes(len(texts) * 384 * 2))
            data = dataset(
                cache.parent,
                [["short_dt", "events", "val", [], "reg"]],
                "train",
                {"batch": 2, "context": 16, "width": 8},
                shuffle_seed=0,
            )
            batch = data[0]
            dt_vals = batch["datetime_values"][batch["sem_types"] == 2].float()
            self.assertGreater(dt_vals.numel(), 0)
            self.assertTrue(torch.isfinite(dt_vals).all())
            # Exact population z-scores of two points are -1.0 and +1.0 (not +/-1e12).
            self.assertAlmostEqual(dt_vals.abs().max().item(), 1.0, places=2)

    def test_rel_event_user_attendance_remains_regression(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            save(raw / "db" / "users.parquet", {"id": [0, 1], "age": [20.0, 30.0]}, pk="id")
            for split, vals in (("train", [1.5, 4.5]), ("val", [2.0, 5.0]), ("test", [3.0, 6.0])):
                save(
                    raw / "tasks" / "user-attendance" / f"{split}.parquet",
                    {"index": [0, 1], "user": [0, 1], "date": dates([1, 2]), "target": vals},
                    fk={"user": "users"},
                    dated=True,
                )
            cache = root / "pre" / "rel-event"
            subprocess.run(
                [
                    str(self.native),
                    "pre",
                    "rel-event",
                    "--input-dir",
                    str(raw),
                    "--output-dir",
                    str(cache),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            texts = json.loads((cache / "text.json").read_text())
            (cache / f"text_emb_{MINILM}.bin").write_bytes(bytes(len(texts) * 384 * 2))
            data = dataset(
                cache.parent,
                [["rel-event", "user-attendance", "target", [], "reg"]],
                "val",
                {"batch": 2, "context": 16, "width": 8},
                shuffle_seed=0,
                forecast_policies={
                    "rel-event": {
                        "user-attendance": {
                            "val": (False, DAY, unix(10)),
                            "test": (True, DAY, unix(20)),
                        }
                    }
                },
            )
            batch = data[0]
            # SemType::Number == 0 (previously broken by casting target to Boolean == 3).
            self.assertTrue((batch["sem_types"][batch["is_targets"]] == 0).all())

    def test_chunked_text_embeddings_match_single_shot(self):
        from rt.experiment_data import write_text_embeddings

        # Encode deterministic values tied to text string so chunking order is verified.
        class TextKeyedModel:
            def encode(self, texts, batch_size, convert_to_tensor, show_progress_bar):
                return torch.stack(
                    [torch.full((384,), float(int(t)), dtype=torch.float32) for t in texts],
                    dim=0,
                )

        texts = [str(i) for i in range(11)]
        model = TextKeyedModel()
        with tempfile.TemporaryDirectory() as tmp:
            single = Path(tmp) / "single.bin"
            chunked = Path(tmp) / "chunked.bin"
            write_text_embeddings(
                model, texts, single, device_type="cpu", name="t", chunk_size=100, batch_size=16
            )
            write_text_embeddings(
                model, texts, chunked, device_type="cpu", name="t", chunk_size=3, batch_size=2
            )
            self.assertEqual(single.read_bytes(), chunked.read_bytes())
            self.assertEqual(len(chunked.read_bytes()), len(texts) * 384 * 2)
