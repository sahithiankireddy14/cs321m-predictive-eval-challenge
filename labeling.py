
from __future__ import annotations

import json
from pathlib import Path

current_path = Path(__file__).parent

with open(current_path / "means_exp10.json") as f:
    _m = json.load(f)

_SUBJ_MEAN       = _m["subject"]
_SUBJ_BENCH      = _m["subject_benchmark"]
_SUBJ_BENCH_COND = _m["subject_bench_cond"]


def _parse_name(sc: str) -> str:
    return sc.strip().splitlines()[0].replace("Name:", "").strip()


def acquisition_function(input: dict) -> float:
    name    = _parse_name(input["subject_content"])
    key_sbc = f"{name}|||{input['benchmark']}|||{input['condition']}"
    key_sb  = f"{name}|||{input['benchmark']}"

    if name not in _SUBJ_MEAN:
        return 5.0
    n_sbc = _SUBJ_BENCH_COND.get(key_sbc) is None and 0 or 1
    n_sb  = _SUBJ_BENCH.get(key_sb) is None and 0 or 1
    if n_sb == 0:
        return 4.0
    if n_sbc == 0:
        return 3.0
    return 1.0
