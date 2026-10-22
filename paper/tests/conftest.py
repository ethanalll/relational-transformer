from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
import yaml


@pytest.fixture
def raw_database(tmp_path):
    def create(name="raw", shift=0.0):
        root = tmp_path / name
        (root / "db").mkdir(parents=True)
        pd.DataFrame(
            dict(
                row_idx=np.arange(4),
                value=np.arange(4, dtype=float) + shift,
                flag=[False, True, False, True],
                always_true=[True] * 4,
                always_false=[False] * 4,
                constant=[1.0] * 4,
                missing=[np.nan, 1.0, 2.0, 3.0],
            )
        ).to_parquet(root / "db" / "users.parquet", index=False)
        pd.DataFrame(
            dict(
                row_idx=np.arange(4),
                user_id=pd.array([0, None, 2, 1], dtype="Int64"),
                value=[0.1, 0.5, 0.9, 1.4],
            )
        ).to_parquet(root / "db" / "events.parquet", index=False)
        manifest = dict(
            name=name,
            tables=dict(
                users=dict(pkey="row_idx", fkeys={}, time_col=None),
                events=dict(pkey="row_idx", fkeys={"user_id": "users"}, time_col=None),
            ),
        )
        (root / "manifest.yaml").write_text(yaml.safe_dump(manifest))
        return root

    return create
