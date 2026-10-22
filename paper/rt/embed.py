"""MiniLM embeddings; deterministic diagnostic embeddings require explicit opt-in."""

import hashlib
import json
from pathlib import Path
import numpy as np
import torch

EMBEDDING_MODEL = "all-MiniLM-L12-v2"


def embed_database(pre_dir, db_name, *, device="cpu", diagnostic=False, batch_size=256):
    root = Path(pre_dir) / db_name
    texts = json.loads((root / "text.json").read_text())
    if diagnostic:
        vectors = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            vectors.append(rng.standard_normal(384).astype(np.float32))
        embeddings = torch.from_numpy(np.stack(vectors))
    else:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            "sentence-transformers/" + EMBEDDING_MODEL,
            device=device,
            model_kwargs={
                "dtype": torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
            },
        )
        embeddings = (
            model.encode(texts, batch_size=batch_size, convert_to_tensor=True).cpu().float()
        )
    if embeddings.shape != (len(texts), 384) or not torch.isfinite(embeddings).all():
        raise ValueError("Invalid embedding matrix")
    embeddings = embeddings.to(torch.bfloat16).contiguous()
    torch.save(embeddings, root / "embeddings.pt")
    (root / f"text_emb_{EMBEDDING_MODEL}.bin").write_bytes(
        embeddings.view(torch.uint16).numpy().tobytes()
    )
    meta_path = root / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta.update(embedding_model=EMBEDDING_MODEL, diagnostic_embeddings=diagnostic, d_text=384)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
