"""CPU end-to-end diagnostic using generated corpora and the full paper model.

Does NOT claim to test the published weights, MiniLM, CUDA, or native sampler.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from rt.corpus import export_legacy_corpus
from rt.preprocess import preprocess_database
from rt.embed import embed_database
from rt.tasks import load_tasks
from rt.data import RelationalDataset
from rt.model import RelationalTransformer
from rt.main import PAPER_DIMS, run
from rt.checkpoints import load_paper_checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plurel-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    sys.path.insert(0, str(Path(args.plurel_root).resolve()))
    from plurel import Config, Choices, DatabaseParams, SCMParams, RSCMParams
    from scripts.synthetic_gen import main as generate
    from torch_frame import stype

    torch.set_num_threads(2)
    torch.manual_seed(42)
    initial = RelationalTransformer(**PAPER_DIMS)
    initial_path = output / "initial_diagnostic.pt"
    torch.save(initial.state_dict(), initial_path)
    del initial
    reports = {}
    for mode in ["legacy_scm", "global_rscm"]:
        cfg = Config(
            generation_mode=mode,
            database_params=DatabaseParams(
                table_layout_choices=Choices("set", ["RandomTree"]),
                num_tables_choices=Choices("set", [3]),
                num_cols_choices=Choices("range", [4, 4]),
                num_rows_entity_table_choices=Choices("set", [12]),
                num_rows_activity_table_choices=Choices("set", [18]),
                col_transform_choices=Choices("set", ["standardize"]),
                column_nan_perc_choices=Choices("set", [0.05]),
                zero_inflation_col_prob=0,
                bool_col_prob=0.5,
            ),
            scm_params=SCMParams(
                scm_layout_choices=Choices("set", ["ErdosRenyi"]),
                col_stype_choices=Choices("set", [stype.numerical]),
                mechanism_type_choices=Choices("set", ["mlp"]),
                mlp_num_layers_choices=Choices("set", [2]),
                activation_choices=Choices("set", [torch.tanh]),
                initialization_choices=Choices("set", [torch.nn.init.xavier_uniform_]),
                source_gen_type_choices=Choices("set", ["gaussian"]),
                propagation_agg_choices=Choices("set", ["sum"]),
                node_noise_type_choices=Choices("set", ["gaussian"]),
                fk_null_prob=0.3,
            ),
            rscm=RSCMParams(max_hops=2, parent_policy="bounded", cell_parent_cap_per_edge=8),
        )
        for split, offset, count in [("train", 20000, 2), ("val", 30000, 1)]:
            corpus = generate(
                offset, count, 1, db_prefix=f"{mode}-{split}", config=cfg, output_dir=output / "raw"
            )
            raw = Path(corpus["databases"][0]["path"]).parent
            exports = output / "exports" / mode / split
            records = export_legacy_corpus(raw, exports)
            pre = output / "pre" / mode / split
            for record in records:
                preprocess_database(exports / record["name"], pre, backend="portable")
                embed_database(pre, record["name"], diagnostic=True)
        train_pre = output / "pre" / mode / "train"
        val_pre = output / "pre" / mode / "val"
        tasks = load_tasks(train_pre)
        print(
            f"{mode}: {len(tasks)} tasks; kinds={sorted({t.task_type for t in tasks})}", flush=True
        )
        report = run(
            dict(
                train_pre_dir=str(train_pre),
                val_pre_dir=str(val_pre),
                output_dir=str(output / "runs" / mode),
                checkpoint=str(initial_path),
                initialization="pretrained",
                backend="portable",
                diagnostic=True,
                steps=args.steps,
                batch_size=2,
                ctx_len=32,
                bfs_width=8,
                lr=1e-4,
                seed=0,
                device="cpu",
                eval_every=args.steps,
                eval_batches=1,
                torch_threads=2,
            )
        )
        model = RelationalTransformer(**PAPER_DIMS)
        load_paper_checkpoint(output / "runs" / mode / "model.pt", model)
        state = {k: v.clone() for k, v in model.state_dict().items()}
        load_paper_checkpoint(output / "runs" / mode / "resume.pt", model)
        if not all(torch.equal(v, model.state_dict()[k]) for k, v in state.items()):
            raise AssertionError("Checkpoint reload changed weights")
        initial_state = torch.load(initial_path, weights_only=True)
        report["parameter_tensors_updated"] = sum(
            not torch.equal(v, initial_state[k]) for k, v in state.items()
        )
        report["checkpoint_reload_exact"] = True
        batch = RelationalDataset(
            tasks, train_pre, batch_size=2, ctx_len=32, max_bfs_width=8, backend="portable"
        )[0]
        model.eval()
        with torch.inference_mode():
            loss, _ = model(batch)
        if not torch.isfinite(loss):
            raise AssertionError("Reloaded loss is nonfinite")
        report["reloaded_loss"] = float(loss)
        reports[mode] = report
        del model, state, initial_state
    result = dict(
        scope="CPU diagnostic, full paper model, generated data",
        reports=reports,
        published_checkpoint_tested=False,
        minilm_tested=False,
        native_sampler_tested=False,
        cuda_tested=False,
    )
    (output / "smoke-report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
