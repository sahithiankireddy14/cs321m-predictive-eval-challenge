
from __future__ import annotations


from datasets import Features, Value, load_dataset
from huggingface_hub import HfApi

REPO_ID = "aims-foundations/measurement-db"
REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}

RESPONSE_FEATURES = Features({
    "subject_id":    Value("string"),
    "item_id":       Value("string"),
    "benchmark_id":  Value("string"),
    "trial":         Value("int64"),
    "test_condition": Value("string"),
    "response":      Value("float64"),
    "correct_answer": Value("string"),
    "trace":         Value("string"),
})


def load_raw_tables():
    repo_files = HfApi().list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    response_files = sorted(
        name for name in repo_files
        if name.endswith(".parquet")
        and name not in REGISTRY_FILES
        and not name.endswith("_traces.parquet")
    )
    responses  = load_dataset(REPO_ID, data_files=response_files,
                              features=RESPONSE_FEATURES, split="train")
    items      = load_dataset(REPO_ID, data_files="items.parquet",      split="train")
    subjects   = load_dataset(REPO_ID, data_files="subjects.parquet",   split="train")
    benchmarks = load_dataset(REPO_ID, data_files="benchmarks.parquet", split="train")
    return responses, items, subjects, benchmarks


def build_examples(responses, items, subjects, benchmarks) -> list[dict]:
    items_by_id      = {r["item_id"]: r      for r in items}
    subjects_by_id   = {r["subject_id"]: r   for r in subjects}
    benchmarks_by_id = {r["benchmark_id"]: r for r in benchmarks}

    examples = []
    for row in responses:
        item      = items_by_id.get(row["item_id"], {})
        subject   = subjects_by_id.get(row["subject_id"], {})
        benchmark = benchmarks_by_id.get(row["benchmark_id"], {})

        content = item.get("content", "")
        if not content:
            continue
        label = row["response"]
        if label not in (0.0, 1.0):
            continue

        examples.append({
            "benchmark":       benchmark.get("benchmark_id") or row["benchmark_id"],
            "condition":       row["test_condition"] or "none",
            "subject_content": render_subject(subject, row["subject_id"]),
            "item_content":    content,
            "label":           int(label),
        })
    return examples


def render_subject(subject: dict, fallback: str) -> str:
    name = subject.get("display_name") or fallback
    lines = [f"Name: {name}"]
    for key, label in [("provider", "Organization"), ("params", "Parameters"),
                       ("release_date", "Released"), ("family", "Family")]:
        v = subject.get(key)
        if v:
            lines.append(f"{label}: {v}")
    return "\n".join(lines)


def parse_name(subject_content: str) -> str:
    first = subject_content.strip().splitlines()[0]
    return first.replace("Name:", "").strip()


def load_examples() -> list[dict]:
    responses, items, subjects, benchmarks = load_raw_tables()
    return build_examples(responses, items, subjects, benchmarks)




def build_means(examples: list[dict]) -> dict:
    import collections
    import numpy as np

    subj_bench = collections.defaultdict(list)
    subj_all   = collections.defaultdict(list)
    for e in examples:
        name = parse_name(e["subject_content"])
        subj_bench[(name, e["benchmark"])].append(e["label"])
        subj_all[name].append(e["label"])

    global_mean = float(np.mean([e["label"] for e in examples]))
    return {
        "global":            global_mean,
        "subject":           {k: float(np.mean(v)) for k, v in subj_all.items()},
        "subject_benchmark": {f"{k[0]}|||{k[1]}": float(np.mean(v)) for k, v in subj_bench.items()},
    }


def encode_examples(examples: list[dict], encoder, batch_size: int = 512):
    import numpy as np

    unique_subjects = list({e["subject_content"] for e in examples})
    unique_items    = list({e["item_content"]    for e in examples})
    print(f"  {len(unique_subjects)} unique subjects, {len(unique_items)} unique items")

    subj_emb = dict(zip(unique_subjects,
        encoder.encode(unique_subjects, batch_size=batch_size, show_progress_bar=True)))
    item_emb = dict(zip(unique_items,
        encoder.encode(unique_items,    batch_size=batch_size, show_progress_bar=True)))

    X = np.array([
        np.concatenate([subj_emb[e["subject_content"]], item_emb[e["item_content"]]])
        for e in examples
    ], dtype=np.float32)
    y = np.array([e["label"] for e in examples], dtype=np.float32)
    return X, y


def train_mlp(X, y, D: int = 384, epochs: int = 10, batch_size: int = 512, device: str = "cpu"):
    import numpy as np
    import torch
    import torch.nn as nn

    perm  = np.random.permutation(len(X))
    split = int(0.9 * len(X))
    train_idx, val_idx = perm[:split], perm[split:]

    X_tr = torch.tensor(X[train_idx]).to(device)
    y_tr = torch.tensor(y[train_idx]).unsqueeze(1).to(device)
    X_va = torch.tensor(X[val_idx]).to(device)
    y_va = torch.tensor(y[val_idx]).unsqueeze(1).to(device)

    mlp = nn.Sequential(
        nn.Linear(2 * D, 256), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(256, 128),   nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(128, 1),
    ).to(device)

    opt     = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    best_val_loss, best_state = float("inf"), None

    for epoch in range(epochs):
        mlp.train()
        perm_t     = torch.randperm(len(X_tr))
        train_loss = 0.0
        for i in range(0, len(X_tr), batch_size):
            idx_b = perm_t[i : i + batch_size]
            opt.zero_grad()
            loss = loss_fn(mlp(X_tr[idx_b]), y_tr[idx_b])
            loss.backward()
            opt.step()
            train_loss += loss.item()
        mlp.eval()
        with torch.no_grad():
            val_loss = loss_fn(mlp(X_va), y_va).item()
        n_batches = max(1, len(X_tr) // batch_size)
        print(f"  epoch {epoch+1}/{epochs}  train={train_loss/n_batches:.4f}  val={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}

    return best_state, best_val_loss
