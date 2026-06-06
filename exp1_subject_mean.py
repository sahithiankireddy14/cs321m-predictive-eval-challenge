
# calculates avg on training set items per subject (model)
import collections, json
import numpy as np

from utils import load_examples, parse_name

print("Loading data...")
examples = load_examples()
print(f"  {len(examples)} examples loaded")

subj_bench = collections.defaultdict(list)
subj_all   = collections.defaultdict(list)

for e in examples:
    name = parse_name(e["subject_content"])
    subj_bench[(name, e["benchmark"])].append(e["label"])
    subj_all[name].append(e["label"])

global_mean = float(np.mean([e["label"] for e in examples]))

means = {
    "global":           global_mean,
    "subject":          {k: float(np.mean(v)) for k, v in subj_all.items()},
    "subject_benchmark": {f"{k[0]}|||{k[1]}": float(np.mean(v)) for k, v in subj_bench.items()},
}

with open("means.json", "w") as f:
    json.dump(means, f, indent=2)

print(f"Saved means.json  (global mean = {global_mean:.3f}, "
      f"{len(means['subject'])} subjects, "
      f"{len(means['subject_benchmark'])} subject-benchmark pairs)")
