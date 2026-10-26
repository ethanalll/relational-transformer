"""Replay an extreme regression task with label statistics and precision checks."""

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    home = Path.home() / 'rt-data'
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pre-dir', type=Path, default=home / 'test-v1/pre/global_rscm')
    ap.add_argument('--db', default='global_rscm-test-300028')
    ap.add_argument('--table', default='table_6')
    ap.add_argument('--column', default='feature_2')
    ap.add_argument('--output-dir', type=Path, default=home / 'test-v1/extreme-case')
    ap.add_argument('--original-checkpoint', type=Path,
                    default=home / 'weights/paper/synthetic-pretrain_rdb_1024_size_4b.pt')
    ap.add_argument('--continued-checkpoint', type=Path, default=home / 'runs/global-seed0/model.pt')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--eval-batches', type=int, default=4)
    ap.add_argument('--ctx-len', type=int, default=1024)
    ap.add_argument('--bfs-width', type=int, default=128)
    ap.add_argument('--eval-seed', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    for key in ['batch_size', 'eval_batches', 'ctx_len', 'bfs_width']:
        if getattr(args, key) < 1:
            ap.error(f'{key} must be positive')

    import numpy as np
    import pyarrow.parquet as pq
    from sklearn.metrics import r2_score
    import torch
    from rt.checkpoints import load_paper_checkpoint
    from rt.corpus import digest
    from rt.data import RelationalDataset
    from rt.main import PAPER_DIMS, move_batch
    from rt.model import RelationalTransformer
    from rt.tasks import load_tasks

    torch.set_num_threads(4)
    if args.output_dir.exists():
        raise FileExistsError(f'{args.output_dir}; choose a fresh --output-dir')
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    matches = [t for t in load_tasks(args.pre_dir)
               if (t.db_name, t.table_name, t.target_column) == (args.db, args.table, args.column)]
    if len(matches) != 1 or matches[0].task_type != 'reg':
        raise ValueError('Expected exactly one matching regression task')
    folder = args.pre_dir / args.db
    meta = json.loads((folder / 'meta.json').read_text())
    if meta.get('backend') != 'native' or meta.get('diagnostic_embeddings') is not False:
        raise ValueError('Native cache with real embeddings required')
    data = RelationalDataset(matches, args.pre_dir, batch_size=args.batch_size,
                             ctx_len=args.ctx_len, max_bfs_width=args.bfs_width,
                             seed=args.eval_seed, backend='native')
    batches = [data[i] for i in range(min(len(data), args.eval_batches))]
    labels = np.concatenate([b['number_values'][b['is_targets']].flatten().float().numpy()
                             for b in batches]).astype(np.float64)
    offset = json.loads((folder/'table_info.json').read_text())[f'{args.table}:Db']['node_idx_offset']
    rows = np.concatenate([b['node_idxs'][b['is_targets']].numpy() - offset for b in batches])

    def stats(values):
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if not len(values):
            return dict(count=0)
        return dict(count=len(values), unique=int(len(np.unique(values))),
                    mean=float(values.mean()), variance=float(values.var()),
                    std=float(values.std()), minimum=float(values.min()), maximum=float(values.max()))

    report = dict(task='/'.join(matches[0].key()), args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  normalized_sample_labels=stats(labels),
                  normalized_sst=float(np.square(labels-labels.mean()).sum()),
                  cache_meta=meta, results={},
                  precision_note='Float32 replay uses the same native caches; cached bfloat16 inputs and quantized saved weights are not restored to pre-quantization precision.')
    print('Sampled normalized labels:', json.dumps(report['normalized_sample_labels']), flush=True)
    record = json.loads((folder/'export.json').read_text())
    raw_path = Path(record['source'])/'db'/f'{args.table}.parquet'
    if raw_path.exists():
        raw = pq.read_table(raw_path, columns=[args.column]).to_pandas()[args.column].to_numpy(dtype=float)
        report['raw_column'] = stats(raw)
        report['raw_sample_labels'] = stats(raw[rows])
        report['raw_column_source'] = str(raw_path)
        print('Full raw column:', json.dumps(report['raw_column']), flush=True)
        print('Sampled raw column:', json.dumps(report['raw_sample_labels']), flush=True)
    else:
        raw = None
        report['raw_column_unavailable'] = str(raw_path)
    args.output_dir.mkdir(parents=True)
    csv_rows = [dict(example=i, physical_row=int(row), normalized_target=float(y),
                     raw_target=float(raw[row]) if raw is not None else None)
                for i,(row,y) in enumerate(zip(rows,labels))]
    with torch.inference_mode():
        for name, checkpoint in [('original',args.original_checkpoint),('continued',args.continued_checkpoint)]:
            model = RelationalTransformer(**PAPER_DIMS)
            load_paper_checkpoint(checkpoint, model)
            report.setdefault('checkpoint_sha256', {})[name] = digest(checkpoint)
            for dtype in [torch.bfloat16, torch.float32]:
                print(f'Running {name}, {dtype}', flush=True)
                # Reload before each precision to avoid rounding the FP32 comparison in place.
                model = model.cpu().float()
                load_paper_checkpoint(checkpoint, model)
                model = model.to(device=device,dtype=dtype).eval()
                predictions, batch_losses = [], []
                for batch in batches:
                    gpu = move_batch(batch,device,dtype)
                    loss,pred = model(gpu)
                    predictions.extend(pred['number'][gpu['is_targets']].flatten().float().cpu().tolist())
                    batch_losses.extend([float(loss)]*int(gpu['is_targets'].sum()))
                predictions = np.asarray(predictions,dtype=np.float64)
                assert np.isfinite(predictions).all()
                errors = predictions-labels
                mse=float(np.square(errors).mean())
                variance=float(labels.var())
                r2=float(r2_score(labels,predictions)) if variance > 0 else None
                key=f'{name}_{str(dtype).split(".")[-1]}'
                result=dict(predictions=stats(predictions), mse=mse, rmse=mse**.5,
                            mae=float(np.abs(errors).mean()), bias=float(errors.mean()),
                            mean_baseline_mse=variance, r2=r2,
                            manual_r2=1-mse/variance if variance>0 else None,
                            huber_float64=float(np.where(np.abs(errors)<=1,.5*errors**2,np.abs(errors)-.5).mean()),
                            model_loss=float(np.mean(batch_losses)))
                report['results'][key]=result
                for row, prediction in zip(csv_rows,predictions):
                    row[key]=float(prediction)
                print(key,json.dumps(result),flush=True)
            del model
            if device.type=='cuda':
                torch.cuda.empty_cache()
    with (args.output_dir/'predictions.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'Saved diagnostics to {args.output_dir}',flush=True)


if __name__=='__main__':
    main()
