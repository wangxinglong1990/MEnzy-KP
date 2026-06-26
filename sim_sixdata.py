import os; os.environ["INFRA_PROVIDER"] = "True"
import sys; sys.path.insert(0,".")
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from src.features.extractor import _get_protein_encoder

with open("textdocs/sixdata.text") as f:
    content = f.read()
six_entries = []
for line in content.strip().split("\n"):
    if not line.strip(): continue
    parts = line.split("：", 1) if "：" in line else line.split(":", 1)
    if len(parts) == 2:
        six_entries.append({"seq": parts[1].strip()})

df5k = pd.read_csv("datafor/orig5000.csv")
six_seqs = [e["seq"] for e in six_entries]

print("ESMC encoding %d seqs..." % (6 + len(df5k)), flush=True)
encoder = _get_protein_encoder()
all_seqs = six_seqs + df5k["Sequence"].tolist()
emb = StandardScaler().fit_transform(encoder.encode(all_seqs).astype(np.float32))
sim = cosine_similarity(emb[6:], emb[:6]).max(axis=1)

print("Similarity distribution:", flush=True)
for p in [0, 10, 25, 50, 75, 90, 95, 99]:
    print("  p%02d: %.4f" % (p, np.percentile(sim, p)), flush=True)
for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
    print("  sim>=%.2f: %d seqs" % (t, (sim >= t).sum()), flush=True)
print("DONE", flush=True)
