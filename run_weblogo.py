"""WebLogo from NCBI peptidase sequences"""
import sys, os, random; sys.path.insert(0,".")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker
from Bio import SeqIO
from io import StringIO
import subprocess, tempfile

df = pd.read_csv("datafor/ncbi5000_final.csv")
seqs = df["Enzyme"].tolist()

# Sample 500 + always include sixdata
six_idx = [i for i,e in enumerate(df["Entry"]) if any(s in str(e) for s in ["AcAP","TcAP","EaAP","KoAP","MnAP","MsAP"])]
random.seed(42)
sample_idx = set(six_idx)
sample_idx.update(random.sample([i for i in range(len(seqs)) if i not in six_idx], 494))
sampled = [seqs[i] for i in sample_idx]
print(f"Sampled {len(sampled)} sequences", flush=True)

# Write FASTA
fasta_str = "\n".join([f">seq{i}\n{s}" for i, s in enumerate(sampled)])
fasta_path = "datafor/sampled_500.fasta"
with open(fasta_path, "w") as f:
    f.write(fasta_str)

# Align with MAFFT
print("Running MAFFT...", flush=True)
aligned = subprocess.run(["mafft", "--auto", "--quiet", fasta_path], capture_output=True, text=True)
if aligned.returncode != 0:
    print("MAFFT error:", aligned.stderr)
    sys.exit(1)
print("Alignment done", flush=True)

# Parse alignment
records = list(SeqIO.parse(StringIO(aligned.stdout), "fasta"))
ali = np.array([list(str(r.seq)) for r in records])
print(f"Alignment: {ali.shape}", flush=True)

# Compute frequency matrix
amino_acids = "ACDEFGHIKLMNPQRSTVWY-"
freq = np.zeros((ali.shape[1], len(amino_acids)))
for j in range(ali.shape[1]):
    col = ali[:, j]
    total = len(col) - np.sum(col == '-')
    for k, aa in enumerate(amino_acids):
        if aa != '-':
            freq[j, k] = np.sum(col == aa) / max(total, 1)

# Information content
bg = np.ones(20) / 20  # uniform background
info = np.zeros((ali.shape[1], 20))
for j in range(ali.shape[1]):
    for k, aa in enumerate(amino_acids[:20]):
        if freq[j, k] > 0:
            info[j, k] = freq[j, k] * np.log2(freq[j, k] / bg[k] + 1e-12)
info = np.maximum(info, 0)
total_bits = info.sum(axis=1)

# Plot two conserved regions (pick top 2 windows)
# Find the two most conserved windows of ~25 positions
window = 25
scores = np.array([total_bits[i:i+window].sum() for i in range(len(total_bits)-window)])
top_idx = np.argsort(scores)[::-1]
# Pick 2 non-overlapping windows
used = set()
windows = []
for idx in top_idx:
    if not any(abs(idx - w) < window for w, _ in windows):
        windows.append(idx)
    if len(windows) == 2:
        break
windows.sort()

print(f"Logo windows: {windows}", flush=True)

fig, axes = plt.subplots(2, 1, figsize=(14, 7))
fig.patch.set_facecolor("white")

colors = {
    'A':'#000000','C':'#000000','F':'#000000','I':'#000000','L':'#000000',
    'M':'#000000','P':'#000000','V':'#000000','W':'#000000',
    'G':'#2ca02c','S':'#2ca02c','T':'#2ca02c','Y':'#2ca02c','H':'#2ca02c',
    'R':'#1f77b4','K':'#1f77b4','D':'#1f77b4','E':'#1f77b4',
    'N':'#9467bd','Q':'#9467bd',
}

for wi, (ax, w_start) in enumerate(zip(axes, windows)):
    w_end = w_start + window
    sub = info[w_start:w_end]

    logo_df = pd.DataFrame(sub, columns=list(amino_acids[:20]))
    logo_df.index = range(w_start+1, w_end+1)

    logo = logomaker.Logo(logo_df, ax=ax, color_scheme=colors, width=0.9)
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    logo.ax.set_ylabel("Information Content (bits)", fontsize=13)
    logo.ax.set_xlabel("Amino Acid Position", fontsize=13)
    logo.ax.set_title(f"Conserved Region {wi+1} (positions {w_start+1}–{w_end})", fontsize=14, fontweight="bold")
    logo.ax.set_ylim(0, 4.2)

plt.tight_layout(pad=3)
out_pdf = "outputs/run_best/weblogo.pdf"
out_png = "outputs/run_best/weblogo.png"
plt.savefig(out_pdf, dpi=300, facecolor="white", edgecolor="none")
plt.savefig(out_png, dpi=200, facecolor="white", edgecolor="none")
plt.close()
print(f"DONE: {out_pdf}", flush=True)
