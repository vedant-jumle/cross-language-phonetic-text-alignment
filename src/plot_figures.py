"""
Generate all figures and tables for the report.
Run from project/ directory:
    python src/plot_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = "report-latex/figures"
os.makedirs(OUT, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────

def load(name):
    with open(f"results/{name}.json") as f:
        return json.load(f)

lev   = load("levenshtein_v2")
sdx   = load("soundex_v2")
bm25  = load("bm25_v2")
trans = load("transliterate_v2")
model = load("model_v2")
faiss = load("faiss_ablation")

RETRIEVERS = [
    ("Levenshtein", lev),
    ("Soundex",     sdx),
    ("BM25",        bm25),
    ("Transliterate", trans),
    ("Byte Encoder (ours)", model),
]

# ── Helper ─────────────────────────────────────────────────────────────────────

def fmt(v, decimals=3):
    return f"{v:.{decimals}f}"

def bold(s, is_bold):
    return r"\textbf{" + s + "}" if is_bold else s

# ── Table 1: Overall results ───────────────────────────────────────────────────

lines = []
lines.append(r"\begin{table}")
lines.append(r"\caption{Overall retrieval performance (11,974 entities, 470,772 queries).}")
lines.append(r"\label{tab:overall}")
lines.append(r"\begin{tabular}{lrrrrr}")
lines.append(r"\toprule")
lines.append(r"Retriever & MRR & R@1 & R@5 & R@10 & NDCG@10 \\")
lines.append(r"\midrule")

for name, d in RETRIEVERS:
    o = d["overall"]
    is_model = name.startswith("Byte")
    row = " & ".join([
        bold(name, is_model),
        bold(fmt(o["MRR"]),       is_model),
        bold(fmt(o["Recall@1"]),  is_model),
        bold(fmt(o["Recall@5"]),  is_model),
        bold(fmt(o["Recall@10"]), is_model),
        bold(fmt(o["NDCG@10"]),   is_model),
    ]) + r" \\"
    lines.append(row)

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

with open(f"{OUT}/table_overall.tex", "w") as f:
    f.write("\n".join(lines) + "\n")
print("✓ table_overall.tex")

# ── Table 2: By query type ─────────────────────────────────────────────────────

lines = []
lines.append(r"\begin{table}")
lines.append(r"\caption{MRR by query type. Combined queries (Latin variant $\rightarrow$ non-Latin script) are the hardest.}")
lines.append(r"\label{tab:by_type}")
lines.append(r"\begin{tabular}{lrrr}")
lines.append(r"\toprule")
lines.append(r"Retriever & Phonetic & Script & Combined \\")
lines.append(r"\midrule")

for name, d in RETRIEVERS:
    t = d["by_type"]
    is_model = name.startswith("Byte")
    row = " & ".join([
        bold(name, is_model),
        bold(fmt(t["phonetic"]["MRR"]),  is_model),
        bold(fmt(t["script"]["MRR"]),    is_model),
        bold(fmt(t["combined"]["MRR"]),  is_model),
    ]) + r" \\"
    lines.append(row)

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

with open(f"{OUT}/table_by_type.tex", "w") as f:
    f.write("\n".join(lines) + "\n")
print("✓ table_by_type.tex")

# ── Table 3: FAISS ablation ────────────────────────────────────────────────────

INDEX_NAMES = {
    "flat_ip":  "FlatIP (exact)",
    "ivf_flat": "IVF-Flat",
    "hnsw":     "HNSW",
    "ivf_pq":   "IVF-PQ",
}
BEST = "hnsw"

lines = []
lines.append(r"\begin{table}")
lines.append(r"\caption{FAISS index ablation on the same model checkpoint. HNSW matches exact search recall at 5.3$\times$ lower latency.}")
lines.append(r"\label{tab:faiss}")
lines.append(r"\begin{tabular}{lrrrr}")
lines.append(r"\toprule")
lines.append(r"Index & R@1 & R@10 & Latency (ms) & Size \\")
lines.append(r"\midrule")

for key, display in INDEX_NAMES.items():
    r = faiss["results"][key]
    size_mb = r["index_size_bytes"] / 1e6
    lat = r["latency_ms"]["mean"]
    is_best = key == BEST
    size_str = f"{size_mb:.1f}\\,MB"
    row = " & ".join([
        bold(display, is_best),
        bold(fmt(r["Recall@1"]),  is_best),
        bold(fmt(r["Recall@10"]), is_best),
        bold(f"{lat:.2f}",        is_best),
        bold(size_str,            is_best),
    ]) + r" \\"
    lines.append(row)

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

with open(f"{OUT}/table_faiss.tex", "w") as f:
    f.write("\n".join(lines) + "\n")
print("✓ table_faiss.tex")

# ── Figure 1: Script gap bar chart ────────────────────────────────────────────

scripts_non_latin = ["ar", "ru", "zh", "he", "hi", "el", "ko", "ja"]

retriever_labels = ["Levenshtein", "Soundex", "BM25", "Transliterate", "Byte Encoder\n(ours)"]
latin_r10  = []
nonlat_r10 = []

for _, d in RETRIEVERS:
    bs = d["by_script"]
    latin_r10.append(bs["latin"]["Recall@10"])
    avg_nl = np.mean([bs[s]["Recall@10"] for s in scripts_non_latin])
    nonlat_r10.append(avg_nl)

x = np.arange(len(RETRIEVERS))
width = 0.35

fig, ax = plt.subplots(figsize=(5.0, 3.2))

bars_lat = ax.bar(x - width/2, latin_r10,  width, label="Latin (phonetic)",    color="#2166ac", zorder=3)
bars_nl  = ax.bar(x + width/2, nonlat_r10, width, label="Non-Latin (avg)",     color="#d1e5f0", edgecolor="#2166ac", zorder=3)

ax.set_ylabel("Recall@10", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(retriever_labels, fontsize=7.5)
ax.set_ylim(0, 1.05)
ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=8, framealpha=0.9)

# annotate gap for last two bars
for i in [3, 4]:
    gap = latin_r10[i] - nonlat_r10[i]
    ax.annotate(
        f"Δ{gap:.2f}",
        xy=(x[i], max(latin_r10[i], nonlat_r10[i]) + 0.02),
        ha="center", fontsize=7, color="#555555"
    )

fig.tight_layout()
fig.savefig(f"{OUT}/script_gap.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/script_gap.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("✓ script_gap.pdf / .png")

# ── Figure 2: Architecture diagram ────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(3.2, 4.5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

BOX_W  = 0.72
BOX_H  = 0.072
LEFT   = (1 - BOX_W) / 2
ARROW  = 0.028

boxes = [
    (0.88, "Input name\n(any script)"),
    (0.75, "UTF-8 bytes\n(vocab = 256)"),
    (0.62, "Byte embedding\n+ positional encoding"),
    (0.47, "6× Transformer layer\n(pre-norm, 8 heads)"),
    (0.34, "Mean pooling\nover non-pad tokens"),
    (0.21, "L2 normalisation"),
    (0.08, "256-d unit embedding"),
]

COLORS = {
    "Input name\n(any script)":           "#f7f7f7",
    "UTF-8 bytes\n(vocab = 256)":         "#e0eaf5",
    "Byte embedding\n+ positional encoding": "#c6dcf0",
    "6× Transformer layer\n(pre-norm, 8 heads)": "#91bfdb",
    "Mean pooling\nover non-pad tokens":  "#c6dcf0",
    "L2 normalisation":                   "#e0eaf5",
    "256-d unit embedding":               "#4393c3",
}
TEXT_COLORS = {
    "256-d unit embedding": "white",
}

for y_center, label in boxes:
    color = COLORS.get(label, "#e8e8e8")
    facecolor = color
    rect = mpatches.FancyBboxPatch(
        (LEFT, y_center - BOX_H / 2), BOX_W, BOX_H,
        boxstyle="round,pad=0.01",
        facecolor=facecolor, edgecolor="#555555", linewidth=0.8,
        zorder=2,
    )
    ax.add_patch(rect)
    tc = TEXT_COLORS.get(label, "#111111")
    ax.text(0.5, y_center, label, ha="center", va="center",
            fontsize=7.5, color=tc, zorder=3, linespacing=1.4)

# arrows between boxes
for i in range(len(boxes) - 1):
    y_top    = boxes[i][0]    - BOX_H / 2
    y_bottom = boxes[i+1][0]  + BOX_H / 2
    y_mid    = (y_top + y_bottom) / 2
    ax.annotate(
        "", xy=(0.5, y_bottom + 0.002), xytext=(0.5, y_top - 0.002),
        arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.0),
        zorder=4,
    )

fig.tight_layout(pad=0)
fig.savefig(f"{OUT}/architecture.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/architecture.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("✓ architecture.pdf / .png")

print("\nAll outputs written to", OUT)
