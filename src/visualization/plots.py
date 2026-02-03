"""
Publication-quality figures for the benchmark paper.

Figures:
  - Fig 2: AUROC heatmap (models × negative sets)
  - Fig 3: Per-region grouped bar chart
  - Fig 4: Zero-shot vs. LoRA fine-tuned performance
  - Fig 5: Pareto front (efficiency vs. performance)
  - Supplementary: ROC curves
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Publication style
plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

MODEL_ORDER = [
    "DNABERT-2-117M",
    "nucleotide-transformer-v2-500m",
    "nucleotide-transformer-2.5b",
    "hyenadna-large",
    "caduceus-ph",
    "evo-1-131k-base",
]

MODEL_SHORT = {
    "DNABERT-2-117M": "DNABERT-2",
    "nucleotide-transformer-v2-500m": "NT-v2-500M",
    "nucleotide-transformer-2.5b": "NT-v2-2.5B",
    "hyenadna-large": "HyenaDNA",
    "caduceus-ph": "Caduceus",
    "evo-1-131k-base": "Evo-1 (7B)",
}

REGION_ORDER = [
    "splice_proximal",
    "utr_5prime",
    "utr_3prime",
    "promoter",
    "enhancer",
    "deep_intronic",
    "intergenic",
]

NEG_SET_LABELS = {
    "N1": "ClinVar Benign",
    "N2": "gnomAD Common",
    "N3": "Matched Random",
}


def load_results(results_dir: str, pattern: str = "*.json") -> list[dict]:
    """Load all JSON result files from a directory."""
    results = []
    for path in sorted(Path(results_dir).glob(pattern)):
        with open(path) as f:
            results.append(json.load(f))
    return results


def plot_auroc_heatmap(
    results: list[dict],
    output_path: str = "results/fig2_auroc_heatmap.pdf",
    metric: str = "auroc",
):
    """
    Fig 2: Heatmap of AUROC across models × negative sets.
    """
    # Build matrix
    data = {}
    for r in results:
        if r["method"] != "zero_shot":
            continue
        model = MODEL_SHORT.get(r["model"], r["model"])
        neg_set = NEG_SET_LABELS.get(r["negative_set"], r["negative_set"])
        score = r["overall"].get(metric, np.nan)
        data[(model, neg_set)] = score

    models = list(dict.fromkeys(MODEL_SHORT.get(r["model"], r["model"]) for r in results if r["method"] == "zero_shot"))
    neg_sets = list(NEG_SET_LABELS.values())

    matrix = np.full((len(models), len(neg_sets)), np.nan)
    for i, m in enumerate(models):
        for j, n in enumerate(neg_sets):
            matrix[i, j] = data.get((m, n), np.nan)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0.5,
        vmax=1.0,
        xticklabels=neg_sets,
        yticklabels=models,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(f"Zero-shot {metric.upper()} across models and negative sets")
    ax.set_xlabel("Negative Set Strategy")
    ax.set_ylabel("Model")

    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


def plot_region_bars(
    results: list[dict],
    output_path: str = "results/fig3_region_bars.pdf",
    negative_set: str = "N3",
    metric: str = "auroc",
):
    """
    Fig 3: Per-region AUROC grouped bar chart (one bar per model).
    """
    rows = []
    for r in results:
        if r["method"] != "zero_shot" or r["negative_set"] != negative_set:
            continue
        model = MODEL_SHORT.get(r["model"], r["model"])
        for region, metrics in r.get("per_region", {}).items():
            rows.append({
                "Model": model,
                "Region": region,
                metric: metrics.get(metric, np.nan),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No data for region bar plot")
        return

    # Order
    df["Region"] = pd.Categorical(df["Region"], categories=REGION_ORDER, ordered=True)
    df = df.sort_values("Region")

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=df, x="Region", y=metric, hue="Model", ax=ax)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax.set_title(f"Per-region {metric.upper()} (Negative set: {NEG_SET_LABELS.get(negative_set, negative_set)})")
    ax.set_ylabel(metric.upper())
    ax.set_xlabel("Non-coding Region Category")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.xticks(rotation=30, ha="right")

    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


def plot_zeroshot_vs_finetune(
    results: list[dict],
    output_path: str = "results/fig4_zeroshot_vs_finetune.pdf",
    negative_set: str = "N3",
):
    """
    Fig 4: Paired dot plot showing performance gain from LoRA fine-tuning.
    """
    zs = {}
    ft = {}
    for r in results:
        if r["negative_set"] != negative_set:
            continue
        model = MODEL_SHORT.get(r["model"], r["model"])
        if r["method"] == "zero_shot":
            zs[model] = r["overall"]["auroc"]
        elif r["method"] == "lora_finetune":
            val = r["overall"]["auroc"]
            ft[model] = val["mean"] if isinstance(val, dict) else val

    models = [m for m in zs if m in ft]
    if not models:
        print("No matching data for zero-shot vs fine-tune plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(models))
    zs_vals = [zs[m] for m in models]
    ft_vals = [ft[m] for m in models]

    # Connected dots
    for i in range(len(models)):
        ax.plot([x[i], x[i]], [zs_vals[i], ft_vals[i]], "k-", alpha=0.3, linewidth=1)

    ax.scatter(x, zs_vals, s=80, c="steelblue", zorder=5, label="Zero-shot")
    ax.scatter(x, ft_vals, s=80, c="coral", zorder=5, label="LoRA fine-tuned")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_title(f"Performance gain from LoRA fine-tuning ({NEG_SET_LABELS.get(negative_set, negative_set)})")
    ax.legend()
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)

    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


def plot_pareto_front(
    results: list[dict],
    model_params: dict = None,
    output_path: str = "results/fig5_pareto_front.pdf",
    negative_set: str = "N3",
):
    """
    Fig 5: Pareto front — parameter count vs. AUROC.
    """
    if model_params is None:
        model_params = {
            "DNABERT-2": 117e6,
            "NT-v2-500M": 500e6,
            "NT-v2-2.5B": 2.5e9,
            "HyenaDNA": 1.6e9,
            "Caduceus": 200e6,
            "Evo-1 (7B)": 7e9,
        }

    rows = []
    for r in results:
        if r["method"] != "zero_shot" or r["negative_set"] != negative_set:
            continue
        model = MODEL_SHORT.get(r["model"], r["model"])
        params = model_params.get(model, np.nan)
        auroc = r["overall"]["auroc"]
        rows.append({"Model": model, "Parameters": params, "AUROC": auroc})

    df = pd.DataFrame(rows)
    if df.empty:
        print("No data for Pareto front")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(df["Parameters"], df["AUROC"], s=120, c="steelblue", edgecolors="k", zorder=5)

    for _, row in df.iterrows():
        ax.annotate(
            row["Model"],
            (row["Parameters"], row["AUROC"]),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=9,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Number of Parameters (log scale)")
    ax.set_ylabel("AUROC")
    ax.set_title(f"Efficiency frontier ({NEG_SET_LABELS.get(negative_set, negative_set)})")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)

    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


def plot_roc_curves(
    scores_dir: str = "results",
    output_path: str = "results/fig_s1_roc_curves.pdf",
    negative_set: str = "N3",
):
    """
    Supplementary: Overlaid ROC curves for all models.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = plt.cm.Set2(np.linspace(0, 1, len(MODEL_SHORT)))

    for i, (full_name, short_name) in enumerate(MODEL_SHORT.items()):
        score_file = Path(scores_dir) / f"scores_{full_name}_{negative_set}.parquet"
        if not score_file.exists():
            continue

        df = pd.read_parquet(score_file)
        from sklearn.metrics import roc_curve as sk_roc_curve

        fpr, tpr, _ = sk_roc_curve(df["label"], df["score"])
        auc_val = roc_auc_score(df["label"], df["score"])

        ax.plot(fpr, tpr, label=f"{short_name} (AUC={auc_val:.3f})",
                color=colors[i], linewidth=1.5)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves ({NEG_SET_LABELS.get(negative_set, negative_set)})")
    ax.legend(loc="lower right", fontsize=9)

    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")
