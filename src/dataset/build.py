"""Build a local HF dataset of all Hub kernel repos (library=kernels), deep."""

import re
from concurrent.futures import ThreadPoolExecutor
from datasets import Dataset
from huggingface_hub import HfApi

api = HfApi()
VARIANT = re.compile(r"torch(\d+)-cxx(\d+)-(cu\d+|rocm\d+|cpu|metal|xpu)-([^-/]+)-([^-/]+)")


def row(m):
    files = api.list_repo_files(m.id)
    v = {match.groups() for f in files if (match := VARIANT.search(f))}
    return {
        "id": m.id,
        "org": m.id.split("/")[0],
        "downloads": m.downloads or 0,
        "likes": m.likes or 0,
        "created_at": str(m.created_at),
        "last_modified": str(m.last_modified),
        "tags": m.tags or [],
        "n_files": len(files),
        "n_variants": len(v),
        "backends": sorted(
            {"cuda" if c.startswith("cu") else "rocm" if c.startswith("rocm") else c for *_, c, _, _ in v}
        ),
        "torch_versions": sorted({f"torch{t}" for t, *_ in v}),
        "cuda_versions": sorted({c for *_, c, _, _ in v if c.startswith("cu")}),
        "rocm_versions": sorted({c for *_, c, _, _ in v if c.startswith("rocm")}),
        "arches": sorted({a for *_, a, _ in v}),
    }


models = list(api.list_models(filter="kernels", full=True))
with ThreadPoolExecutor(max_workers=24) as ex:
    rows = list(ex.map(row, models))

rows.sort(key=lambda r: r["downloads"], reverse=True)
Dataset.from_list(rows).to_parquet("kernels.parquet")
