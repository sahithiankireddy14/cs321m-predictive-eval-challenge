"""
Experiment 17: item k-NN difficulty estimation (Stage 2a).

For each test item, find K most similar training items by cosine similarity
of sentence embeddings, then use their observed pass rates to estimate
per-item difficulty.

Formula:
    neighbor_difficulty = weighted_avg(pass_rate of K nearest training items)
    item_residual       = neighbor_difficulty - benchmark_mean
    p = subject_bench_accuracy + alpha * item_residual

No neural network training — directly uses observed item difficulties.

Produces: item_knn_data.npz (embeddings + pass rates + benchmark ids)
"""
import modal

app = modal.App("item-knn-exp17")
vol = modal.Volume.from_name("ncf-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "sentence-transformers",
        "datasets", "huggingface_hub", "numpy",
    )
)

MIN_OBS = 5 


@app.function(image=image, gpu="T4", timeout=7200, volumes={"/outputs": vol})
def build():
    import collections, json
    import numpy as np
    import torch
    from datasets import Features, Value, load_dataset
    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer

    REPO_ID        = "aims-foundations/measurement-db"
    REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}

    print("Loading data...")
    repo_files = HfApi().list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    response_files = sorted(
        name for name in repo_files
        if name.endswith(".parquet")
        and name not in REGISTRY_FILES
        and not name.endswith("_traces.parquet")
    )
    response_features = Features({
        "subject_id": Value("string"), "item_id": Value("string"),
        "benchmark_id": Value("string"), "trial": Value("int64"),
        "test_condition": Value("string"), "response": Value("float64"),
        "correct_answer": Value("string"), "trace": Value("string"),
    })
    responses  = load_dataset(REPO_ID, data_files=response_files,
                              features=response_features, split="train")
    items      = load_dataset(REPO_ID, data_files="items.parquet",      split="train")
    benchmarks = load_dataset(REPO_ID, data_files="benchmarks.parquet", split="train")

    items_by_id      = {r["item_id"]: r for r in items}
    benchmarks_by_id = {r["benchmark_id"]: r for r in benchmarks}


    print("Computing per-item pass rates...")
    item_counts  = collections.defaultdict(lambda: [0, 0])
    item_bench   = {}
    item_content = {}

    for row in responses:
        if row["response"] not in (0.0, 1.0):
            continue
        iid = row["item_id"]
        item_counts[iid][0] += int(row["response"])
        item_counts[iid][1] += 1
        if iid not in item_bench:
            bench = benchmarks_by_id.get(row["benchmark_id"], {})
            item_bench[iid]   = bench.get("benchmark_id") or row["benchmark_id"]
            item_content[iid] = items_by_id.get(iid, {}).get("content", "")


    well_observed = [
        iid for iid in item_counts
        if item_counts[iid][1] >= MIN_OBS and item_content.get(iid, "")
    ]
    print(f"  {len(well_observed)} items with >= {MIN_OBS} observations")

  
    bench_correct = collections.defaultdict(lambda: [0, 0])
    for iid in well_observed:
        b = item_bench[iid]
        bench_correct[b][0] += item_counts[iid][0]
        bench_correct[b][1] += item_counts[iid][1]
    bench_mean = {b: v[0]/v[1] for b, v in bench_correct.items() if v[1] > 0}


    item_pass_rate = {
        iid: item_counts[iid][0] / item_counts[iid][1]
        for iid in well_observed
    }
    item_residual = {
        iid: item_pass_rate[iid] - bench_mean.get(item_bench[iid], 0.5)
        for iid in well_observed
    }

    print(f"  residual mean={float(np.mean(list(item_residual.values()))):.3f}  "
          f"std={float(np.std(list(item_residual.values()))):.3f}")

    
    print("Encoding items with all-MiniLM-L6-v2...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

    texts = [item_content[iid] for iid in well_observed]
    embs  = encoder.encode(texts, batch_size=512, show_progress_bar=True,
                           normalize_embeddings=True)  
    print(f"  Embeddings shape: {embs.shape}")


    residuals  = np.array([item_residual[iid]  for iid in well_observed], dtype=np.float32)
    pass_rates = np.array([item_pass_rate[iid] for iid in well_observed], dtype=np.float32)
    bench_ids  = np.array([item_bench[iid]     for iid in well_observed])

    np.savez_compressed(
        "/outputs/item_knn_data.npz",
        embeddings = embs.astype(np.float16),  
        residuals  = residuals,
        pass_rates = pass_rates,
        bench_ids  = bench_ids,
    )

 
    with open("/outputs/item_knn_meta.json", "w") as f:
        json.dump({"bench_mean": bench_mean}, f)

    size_mb = embs.astype(np.float16).nbytes / 1e6
    print(f"  Saved item_knn_data.npz ({size_mb:.1f} MB embeddings, "
          f"{len(well_observed)} items)")

    vol.commit()
    print("Done.")


@app.local_entrypoint()
def main():
    build.remote()
    print("\nDownload with:")
    print("  modal volume get ncf-outputs item_knn_data.npz .")
    print("  modal volume get ncf-outputs item_knn_meta.json .")
