import os; os.environ["INFRA_PROVIDER"] = "True"
import sys; sys.path.insert(0,".")
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from src.features.extractor import _get_protein_encoder

# Parse sixdata
with open("textdocs/sixdata.text") as f:
    content = f.read()
six_entries = []
for line in content.strip().split("\n"):
    if not line.strip(): continue
    parts = line.split("：", 1) if "：" in line else line.split(":", 1)
    if len(parts) == 2:
        six_entries.append({"name": parts[0].strip(), "seq": parts[1].strip()})

# Match to 493 for proper names
df493 = pd.read_excel("datafor/Ajinomoto_psiblast_output.xlsx")
six_full = []
for e in six_entries:
    seq_upper = e["seq"].upper().strip()
    m = df493[df493["Enzyme"].str.upper().str.strip() == seq_upper]
    if len(m) > 0:
        r = m.iloc[0]
        six_full.append({"name": e["name"], "short": e["name"].split(".")[-1],
                         "entry": r["条目"], "seq": e["seq"], "substrate": r["Substrates"]})
    else:
        six_full.append({"name": e["name"], "short": e["name"].split(".")[-1],
                         "entry": e["name"], "seq": e["seq"], "substrate": "COC(=O)CC[NH3+]"})

# Load 5000
df5k = pd.read_csv("datafor/orig5000.csv")

# ESMC
print("ESMC encoding %d seqs..." % (6+len(df5k)), flush=True)
encoder = _get_protein_encoder()
all_seqs = [e["seq"] for e in six_full] + df5k["Sequence"].tolist()
emb = StandardScaler().fit_transform(encoder.encode(all_seqs).astype(np.float32))
sim = cosine_similarity(emb[6:], emb[:6]).max(axis=1)

threshold = 0.15
keep = sim >= threshold
print("Keep %d/%d (sim >= %.2f)" % (keep.sum(), len(df5k), threshold), flush=True)

# Build dataset
rows = []
for e in six_full:
    rows.append({"Entry": e["entry"], "Full_Header": e["entry"],
                 "Enzyme": e["seq"], "Length": len(e["seq"]),
                 "Substrates": e["substrate"]})
for _, r in df5k[keep].iterrows():
    rows.append({"Entry": r["Sequence_ID"], "Full_Header": r["Full_Header"],
                 "Enzyme": r["Sequence"], "Length": r["Length"],
                 "Substrates": "COC(=O)CC[NH3+]"})
df_final = pd.DataFrame(rows)
print("Final: %d seqs" % len(df_final), flush=True)
df_final.to_csv("datafor/final_sixdata_filtered.csv", index=False)

# Predict
import subprocess
print("Predicting...", flush=True)
subprocess.run([sys.executable, "-u", "predict_experiment_csv.py",
    "--input", "datafor/final_sixdata_filtered.csv",
    "--seq-col", "Enzyme", "--smiles-col", "Substrates",
    "--output", "datafor/final_sixdata_filtered.csv"], check=True)

# Report
df = pd.read_csv("datafor/final_sixdata_filtered.csv")
df = df.sort_values("Pred_kcat_over_Km", ascending=False).reset_index(drop=True)
ratio = df["Pred_kcat_over_Km"].values
n_total = len(df)
n_top = int(n_total * 0.10)
cutoff = ratio[n_top - 1]

print("\nSixdata rankings:", flush=True)
for e in six_full:
    sn = e["entry"]
    ss = e["short"]
    m = df["Entry"] == sn
    if m.any():
        r = int(df[m].index[0]) + 1
        v = float(df[m]["Pred_kcat_over_Km"].iloc[0])
        in_top = "GREEN" if r <= n_top else "gray"
        print("  %s (#%d/%d) kcat/Km=%.0f [%s]" % (ss, r, n_total, v, in_top), flush=True)

print("Total: %d  Top10%%: %d  cutoff: %.0f" % (n_total, n_top, cutoff), flush=True)
print("DONE", flush=True)
