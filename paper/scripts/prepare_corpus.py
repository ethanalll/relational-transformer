"""Convert a modern PLuREL corpus for paper RT. Defaults to native + real MiniLM."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rt.corpus import export_legacy_corpus
from rt.preprocess import preprocess_database
from rt.embed import embed_database


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--pre-dir", required=True)
    ap.add_argument("--backend", choices=["native", "portable"], default="native")
    ap.add_argument("--embedding-device", default="cpu")
    ap.add_argument("--diagnostic-embeddings", action="store_true")
    args = ap.parse_args()
    if args.diagnostic_embeddings and args.backend != "portable":
        ap.error("Diagnostic embeddings are only allowed with the portable diagnostic backend")
    records = export_legacy_corpus(args.corpus, args.export_dir)
    for record in records:
        preprocess_database(
            Path(args.export_dir) / record["name"], args.pre_dir, backend=args.backend
        )
        embed_database(
            args.pre_dir,
            record["name"],
            device=args.embedding_device,
            diagnostic=args.diagnostic_embeddings,
        )
    print(
        json.dumps(
            dict(
                pre_dir=str(Path(args.pre_dir).resolve()),
                databases=len(records),
                backend=args.backend,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
