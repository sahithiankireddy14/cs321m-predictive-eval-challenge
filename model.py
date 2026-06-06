"""Exp 17: item k-NN Stage 2a on top of Exp 10 lookup.

Stage 1: subject ability from lookup
Stage 2a: k-NN item difficulty — find K most similar training items,
          use their observed pass rates as difficulty estimate
Stage 3: p = sigmoid(logit(subject_bench_acc) + alpha * item_residual)
       + Platt scaling from adaptive labels
"""
from __future__ import annotations

import json, math
from pathlib import Path
import numpy as np

current_path = Path(__file__).parent


with open(current_path / "means_exp10.json") as f:
    m = json.load(f)

G               = m["global"]
SUBJ_MEAN       = m["subject"]
SUBJ_BENCH      = m["subject_benchmark"]
SUBJ_BENCH_COND = m["subject_bench_cond"]
BC_MEAN         = m["benchmark_condition"]
FAM_MEAN        = m["family"]
COND_DELTA      = m["condition_delta"]
BENCH_BIAS      = m["bench_bias"]
BENCH_STD       = m["bench_std"]


with open(current_path / "item_knn_meta.json") as f:
    knn_meta = json.load(f)
BENCH_MEAN = knn_meta["bench_mean"]

knn = np.load(current_path / "item_knn_data.npz", allow_pickle=True)
KNN_EMBS       = knn["embeddings"].astype(np.float32)
KNN_RESIDUALS  = knn["residuals"]
KNN_BENCH_IDS  = knn["bench_ids"]

K           = 10
ALPHA_BLEND = 0.3


from sentence_transformers import SentenceTransformer
import os

def cache_dir():
    for candidate in [os.environ.get("HF_HOME",""), "/app/hf_cache",
                      str(current_path / ".hf_cache")]:
        if not candidate: continue
        p = Path(candidate)
        try: p.mkdir(parents=True, exist_ok=True)
        except OSError: continue
        if os.access(p, os.W_OK): return str(p)
    return None

ENCODER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                               cache_folder=cache_dir())


def parse_name(sc):
    return sc.strip().splitlines()[0].replace("Name:", "").strip()

def parse_family(sc):
    for line in sc.strip().splitlines():
        if line.startswith("Family:"):
            return line.replace("Family:", "").strip().lower()
    name = parse_name(sc).lower()
    for fam in ["llama","gpt","claude","gemini","qwen","mistral",
                "falcon","phi","gemma","deepseek"]:
        if fam in name: return fam
    return "unknown"

def lookup(sc, benchmark, condition):
    name   = parse_name(sc)
    family = parse_family(sc)
    key1   = f"{name}|||{benchmark}|||{condition}"
    if key1 in SUBJ_BENCH_COND:
        return SUBJ_BENCH_COND[key1]
    key2  = f"{name}|||{benchmark}"
    delta = COND_DELTA.get(f"{benchmark}|||{condition}", 0.0)
    if key2 in SUBJ_BENCH:
        return float(np.clip(SUBJ_BENCH[key2] + delta, 0.01, 0.99))
    if name in SUBJ_MEAN:
        return float(np.clip(SUBJ_MEAN[name] + delta, 0.01, 0.99))
    if family in FAM_MEAN:
        return FAM_MEAN[family]
    key4 = f"{benchmark}|||{condition}"
    if key4 in BC_MEAN:
        return float(BC_MEAN[key4])
    return G


def item_difficulty_residual(item_content: str, benchmark: str) -> float:
    emb = ENCODER.encode(item_content, convert_to_tensor=False,
                         normalize_embeddings=True)
    emb = emb.astype(np.float32)

    sims = KNN_EMBS @ emb
    same_bench = (KNN_BENCH_IDS == benchmark).astype(np.float32)
    boosted    = sims + 0.1 * same_bench

    top_k = np.argpartition(boosted, -K)[-K:]
    top_sims = sims[top_k]

    weights = np.exp(top_sims * 10)
    weights /= weights.sum()

    return float(np.dot(weights, KNN_RESIDUALS[top_k]))


def logit(p):
    return math.log(float(np.clip(p,1e-6,1-1e-6))/(1-float(np.clip(p,1e-6,1-1e-6))))

def sigmoid(x):
    return 1.0/(1.0+math.exp(-x))

def fit_platt(preds, labels, b_init=0.0, reg=1.0):
    if len(preds) < 2: return 1.0, b_init
    logits = [logit(p) for p in preds]
    n = len(preds)
    a, b = 1.0, b_init
    for _ in range(300):
        da, db = 0.0, 0.0
        for li, yi in zip(logits, labels):
            err = sigmoid(a*li+b)-yi
            da += err*li; db += err
        a -= 0.1*(da/n+reg*(a-1.0))
        b -= 0.1*(db/n+reg*(b-b_init))
    return a, b

def build_calibrators(labeled):
    by_bench = {}
    all_p, all_y = [], []
    for d in labeled:
        bench = d["benchmark"]
        p_l   = lookup(d["subject_content"], bench, d["condition"])
        res   = item_difficulty_residual(d["item_content"], bench)
        p     = sigmoid(logit(p_l) + ALPHA_BLEND * res)
        by_bench.setdefault(bench, ([], []))
        by_bench[bench][0].append(p)
        by_bench[bench][1].append(int(d["label"]))
        all_p.append(p); all_y.append(int(d["label"]))
    per_bench = {}
    for bench, (ps, ys) in by_bench.items():
        n      = max(len(ps), 1)
        b_init = logit(G + BENCH_BIAS.get(bench, 0.0)) - logit(G)
        reg    = (BENCH_STD.get(bench, 0.3) * 5.0) / n
        per_bench[bench] = fit_platt(ps, ys, b_init=b_init, reg=reg)
    global_cal = fit_platt(all_p, all_y, reg=1.0/max(len(all_p),1))
    return per_bench, global_cal

per_bench_cal = {}
global_cal    = (1.0, 0.0)
calibrated    = False

def predict(input: dict, labeled: list[dict] | None = None) -> float:
    global per_bench_cal, global_cal, calibrated

    if labeled and not calibrated:
        per_bench_cal, global_cal = build_calibrators(labeled)
        calibrated = True

    p_lookup = lookup(input["subject_content"], input["benchmark"], input["condition"])
    res      = item_difficulty_residual(input["item_content"], input["benchmark"])
    p        = sigmoid(logit(p_lookup) + ALPHA_BLEND * res)

    if calibrated:
        a, b = per_bench_cal.get(input["benchmark"], global_cal)
        p    = sigmoid(a * logit(p) + b)

    return float(np.clip(p, 0.01, 0.99))
