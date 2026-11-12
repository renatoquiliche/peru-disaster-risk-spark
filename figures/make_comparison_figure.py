"""Hold-out (2021) comparison: published XGBoost vs Spark MLlib GBT vs FLAML AutoML.

Every metric is recomputed from each model's confusion matrix with the same formulas,
so the three approaches are compared on identical definitions.

Sources of the confusion matrices
- Published: Figure 14 of the PUC-Rio thesis version of the study (hold-out test set).
- Spark GBT, FLAML: outputs of the notebooks in this repo, run on Databricks serverless.
AUC is taken from the notebooks; the published study does not report it.

All three models were tuned on the same objective, GIBA as the study's code computes it:
imblearn make_index_balanced_accuracy(alpha=0.9, squared=True)(geometric_mean_score), which
equals G-Mean squared = sensitivity x specificity (the dominance term is zero in that call).

Run:  python figures/make_comparison_figure.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# (TP, FN, FP, TN, AUC)
RUNS = {
    "Floods": {
        "Published XGBoost (Bayesian search)": (148, 50, 4858, 23107, None),
        "Spark MLlib GBT (grid search)":       (133, 65, 4720, 23245, 0.8145),
        "FLAML AutoML":                        (153, 45, 4925, 23040, 0.8871),
    },
    "Landslides": {
        "Published XGBoost (Bayesian search)": (1209, 372, 3613, 22969, None),
        "Spark MLlib GBT (grid search)":       (1377, 204, 7093, 19489, 0.8734),
        "FLAML AutoML":                        (1222, 359, 3901, 22681, 0.9116),
    },
}

def metrics(TP, FN, FP, TN, auc):
    tpr, tnr = TP / (TP + FN), TN / (TN + FP)
    gmean = math.sqrt(tpr * tnr)
    mcc = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    prec = TP / (TP + FP)
    return {"AUC": auc, "Sensitivity": tpr, "Specificity": tnr,
            "Balanced accuracy": (tpr + tnr) / 2, "G-Mean": gmean,
            "GIBA": tpr * tnr, "MCC": mcc, "Precision": prec,
            "F1": 2 * prec * tpr / (prec + tpr),
            "Accuracy": (TP + TN) / (TP + TN + FP + FN),
            "TP": TP, "FN": FN, "FP": FP, "TN": TN}

rows = [{"hazard": h, "model": m, **metrics(*v)} for h, d in RUNS.items() for m, v in d.items()]
table = pd.DataFrame(rows)
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
table.round(4).to_csv(os.path.join(ROOT, "results", "holdout_comparison.csv"), index=False)

# ---------------------------------------------------------------- figure
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]            # validated categorical slots 1-3
SHOWN = ["GIBA", "Sensitivity", "Specificity", "G-Mean", "MCC", "Precision", "AUC"]
models = list(RUNS["Floods"])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.2), sharex=True, facecolor=SURFACE)
width = 0.26
for ax, hazard in zip(axes, RUNS):
    ax.set_facecolor(SURFACE)
    sub = table[table.hazard == hazard].set_index("model")
    for j, model in enumerate(models):
        for i, metric in enumerate(SHOWN):
            x = i + (j - 1) * width
            v = sub.loc[model, metric]
            if v is None or pd.isna(v):
                ax.text(x, 0.02, "not\nreported", ha="center", va="bottom", fontsize=7.5,
                        color=MUTED, style="italic")
                continue
            ax.bar(x, v, width - 0.03, color=COLORS[j], label=model if i == 0 else None,
                   zorder=3)
            ax.text(x, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    n_pos = sub.iloc[0][["TP", "FN"]].sum(); n = sub.iloc[0][["TP", "FN", "FP", "TN"]].sum()
    ax.set_title(f"{hazard} — {int(n_pos):,} affected of {int(n):,} households "
                 f"({n_pos / n:.1%})", loc="left", fontsize=11.5, color=INK, pad=8)
    ax.set_ylim(0, 1.08); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(range(len(SHOWN))); ax.set_xticklabels(SHOWN, color=INK, fontsize=10.5)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.06, 0.945), ncol=3,
           frameon=False, fontsize=10, labelcolor=INK, handlelength=1.2, columnspacing=2.2)
fig.suptitle("Hold-out test (2021): published model vs. Spark baseline vs. AutoML",
             x=0.065, y=0.985, ha="left", fontsize=14, color=INK, fontweight="bold")
fig.text(0.065, 0.012,
         "All three models tuned on GIBA, the study's objective (= sensitivity x specificity). Metrics "
         "recomputed from each confusion matrix at a 0.5 threshold. One run per model.",
         fontsize=8.5, color=MUTED)
fig.subplots_adjust(left=0.065, right=0.985, top=0.86, bottom=0.085, hspace=0.28)
out = os.path.join(HERE, "holdout_comparison.png")
fig.savefig(out, dpi=170, facecolor=SURFACE)
print("wrote", out)
print(table[["hazard", "model"] + SHOWN + ["TP", "FN", "FP"]].round(4).to_string(index=False))
