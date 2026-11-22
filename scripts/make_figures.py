#!/usr/bin/env python
"""Build every figure of the paper from the result tables in ``results/``.

Usage::

    python scripts/make_figures.py --results-dir results --figures-dir results/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURES_DIR, MODEL_LABELS, MODEL_ORDER, RESULTS_DIR  # noqa: E402

# --------------------------------------------------------------------------- #
# SBC-friendly style: Times/Helvetica, grayscale-safe, high resolution
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

#: Grayscale-safe palette (works in black-and-white printing).
GRAYS = ["#1a1a1a", "#4d4d4d", "#808080", "#b3b3b3"]
HATCHES = ["", "///", "...", "xxx"]
MARKERS = ["o", "s", "^", "D"]
SHORT_LABELS = {"nb": "NB", "svm": "SVM", "logreg": "LR", "mlp": "MLP"}


def _palette(models: list[str]) -> dict[str, tuple[str, str]]:
    return {model: (GRAYS[i % len(GRAYS)], HATCHES[i % len(HATCHES)])
            for i, model in enumerate(models)}


# --------------------------------------------------------------------------- #
# Figure 1 -- methodological flow chart
# --------------------------------------------------------------------------- #
def figure_pipeline(path: Path) -> None:
    steps = [
        ("Corpus IMDb\n50.000 críticas rotuladas", "#e8e8e8"),
        ("Divisão estratificada\n80 % treino / 20 % teste", "#f2f2f2"),
        ("Pré-processamento\nlimpeza, tokenização, stemming", "#f2f2f2"),
        ("Representação TF-IDF\nunigramas + bigramas, 50 mil termos", "#f2f2f2"),
        ("Busca de hiperparâmetros\nvalidação cruzada 5-fold", "#f2f2f2"),
        ("Treinamento final\nconjunto de treinamento completo", "#f2f2f2"),
        ("Avaliação única\nconjunto de teste isolado", "#f2f2f2"),
        ("Comparação de desempenho\ne custo computacional", "#e8e8e8"),
    ]

    fig, ax = plt.subplots(figsize=(3.15, 5.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(steps))
    ax.axis("off")

    height, gap = 0.72, 0.28
    for index, (text, color) in enumerate(steps):
        y = len(steps) - index - 1
        box = FancyBboxPatch(
            (0.04, y + gap / 2 - 0.02), 0.92, height,
            boxstyle="round,pad=0.012,rounding_size=0.03",
            linewidth=0.7, edgecolor="black", facecolor=color,
        )
        ax.add_patch(box)
        ax.text(0.5, y + gap / 2 + height / 2 - 0.02, text,
                ha="center", va="center", fontsize=7.2, linespacing=1.35)
        if index < len(steps) - 1:
            ax.add_patch(FancyArrowPatch(
                (0.5, y + gap / 2 - 0.03), (0.5, y - gap / 2 + 0.005),
                arrowstyle="-|>", mutation_scale=7, linewidth=0.7, color="black",
            ))
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 2 -- metric comparison across classifiers
# --------------------------------------------------------------------------- #
def figure_metrics(table: pd.DataFrame, path: Path) -> None:
    metrics = [("accuracy", "Acurácia"), ("precision", "Precisão"),
               ("recall", "Recall"), ("f1", "F1-score")]
    models = [m for m in MODEL_ORDER if m in set(table["model"])]
    palette = _palette(models)

    fig, ax = plt.subplots(figsize=(3.4, 2.55))
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    lowest = 100.0

    for index, model in enumerate(models):
        row = table.set_index("model").loc[model]
        values = [100 * row[f"{metric}_mean"] for metric, _ in metrics]
        errors = [100 * row.get(f"{metric}_std", 0.0) for metric, _ in metrics]
        lowest = min(lowest, min(value - error for value, error in zip(values, errors)))
        ax.bar(
            x + (index - (len(models) - 1) / 2) * width, values, width * 0.92,
            yerr=errors, capsize=1.8, label=SHORT_LABELS.get(model, model),
            color=palette[model][0], hatch=palette[model][1],
            edgecolor="black", linewidth=0.4,
            error_kw={"elinewidth": 0.5, "capthick": 0.5},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("Valor (%)", fontsize=8)
    ax.set_ylim(max(0, np.floor((lowest - 1.5) / 5) * 5), 100)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.6)
    ax.legend(ncol=4, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), handlelength=1.0,
              columnspacing=0.6, handletextpad=0.3, fontsize=7)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 3 -- performance vs. training cost
# --------------------------------------------------------------------------- #
def figure_cost(table: pd.DataFrame, path: Path) -> None:
    models = [m for m in MODEL_ORDER if m in set(table["model"])]
    palette = _palette(models)
    data = table.set_index("model").loc[models]

    fig, ax = plt.subplots(figsize=(3.4, 2.45))
    for index, model in enumerate(models):
        row = data.loc[model]
        ax.scatter(row["final_fit_s"], 100 * row["f1"],
                   s=34, color=palette[model][0], edgecolor="black",
                   linewidth=0.5, zorder=3, marker=MARKERS[index % len(MARKERS)],
                   label=MODEL_LABELS[model])
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("Tempo de treinamento (s, escala log)", fontsize=8)
    ax.set_ylabel("F1-score no teste (%)", fontsize=8)
    ax.grid(linestyle=":", linewidth=0.4, alpha=0.6)
    ax.legend(frameon=False, loc="best", fontsize=6.5, handletextpad=0.4,
              borderpad=0.2, labelspacing=0.3)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 4 -- confusion matrices
# --------------------------------------------------------------------------- #
def figure_confusion(table: pd.DataFrame, path: Path) -> None:
    models = [m for m in MODEL_ORDER if m in set(table["model"])]
    data = table.set_index("model").loc[models]
    n = len(models)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4, 1.35 * nrows + 0.25))
    axes = np.atleast_1d(axes).ravel()
    for ax, model in zip(axes, models):
        row = data.loc[model]
        matrix = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=float)
        matrix_pct = 100 * matrix / matrix.sum()
        ax.imshow(matrix_pct, cmap="Greys", vmin=0, vmax=100)
        for (i, j), value in np.ndenumerate(matrix_pct):
            ax.text(j, i, f"{value:.1f}%", ha="center", va="center", fontsize=6,
                    color="white" if value > 55 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Neg.", "Pos."], fontsize=6)
        ax.set_yticklabels(["Neg.", "Pos."], fontsize=6)
        ax.set_title(SHORT_LABELS.get(model, model), fontsize=7, pad=2)
        ax.set_xlabel("Previsto", fontsize=6, labelpad=1)
        ax.set_ylabel("Real", fontsize=6, labelpad=1)
    for ax in axes[len(models):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 5 -- most discriminative terms of the linear SVM
# --------------------------------------------------------------------------- #
def figure_features(path_csv: Path, path: Path, n: int = 12) -> None:
    table = pd.read_csv(path_csv)
    positives = table[table["polarity"] == "positive"].nlargest(n, "weight").iloc[::-1]
    negatives = table[table["polarity"] == "negative"].nsmallest(n, "weight").iloc[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(3.3, 2.3))
    for ax, data, color, title in (
        (axes[0], negatives, "#4d4d4d", "Negativos"),
        (axes[1], positives, "#1a1a1a", "Positivos"),
    ):
        ax.barh(np.arange(len(data)), data["weight"], color=color,
                edgecolor="black", linewidth=0.3, height=0.75)
        ax.set_yticks(np.arange(len(data)))
        ax.set_yticklabels([term.replace("not_", "¬") for term in data["term"]],
                           fontsize=5.6)
        ax.set_title(title, fontsize=7)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.tick_params(axis="x", labelsize=5.5)
        ax.set_xlabel("Peso", fontsize=6, labelpad=1)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args(argv)

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.results_dir / "metrics_test.csv")

    figure_pipeline(args.figures_dir / "fig1_fluxograma.png")
    figure_metrics(table, args.figures_dir / "fig2_metricas.png")
    figure_cost(table, args.figures_dir / "fig4_custo.png")
    figure_confusion(table, args.figures_dir / "fig3_matrizes.png")

    svm_features = args.results_dir / "top_features_svm.csv"
    if svm_features.exists():
        figure_features(svm_features, args.figures_dir / "fig5_termos.png")

    summary = {
        "figures": sorted(p.name for p in args.figures_dir.glob("*.png")),
        "models": table["model"].tolist(),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

