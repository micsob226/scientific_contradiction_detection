import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS = {
    "TF-IDF":              {"ndcg": 0.591, "map": 0.541, "recall": 0.726},
    "BM25":                {"ndcg": 0.706, "map": 0.667, "recall": 0.812},
    "MiniLM":               {"ndcg": 0.655, "map": 0.600, "recall": 0.811},
    "MiniLM+Reranked":      {"ndcg": 0.775, "map": 0.735, "recall": 0.888},
    "MiniLM-FT":            {"ndcg": 0.725, "map": 0.666, "recall": 0.894},
    "MiniLM-FT+Reranked":   {"ndcg": 0.791, "map": 0.753, "recall": 0.898},
    "SPECTER":             {"ndcg": 0.368, "map": 0.306, "recall": 0.560},
    "SPECTER+Reranked":    {"ndcg": 0.674, "map": 0.645, "recall": 0.744},
    "BGE":                 {"ndcg": 0.786, "map": 0.741, "recall": 0.911},
    "BGE+Reranked":        {"ndcg": 0.795, "map": 0.750, "recall": 0.924},
    "BGE-FT":              {"ndcg": 0.830, "map": 0.797, "recall": 0.928},
    "BGE-FT+Reranked":     {"ndcg": 0.798, "map": 0.754, "recall": 0.924},
}

OUTPUT_DIR = Path("slides/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLOR_DEFAULT = "#4878A6"
COLOR_HIGHLIGHT = "#2E7D32"
COLOR_WARNING = "#C0392B"

def plot_main_comparison():
    sorted_methods = sorted(RESULTS.items(), key=lambda x: x[1]["ndcg"], reverse=True)
    names  = [m[0] for m in sorted_methods][::-1]
    values = [m[1]["ndcg"] for m in sorted_methods][::-1]
    colors = [COLOR_HIGHLIGHT if name == "BGE-FT" else COLOR_DEFAULT for name in names]

    fig, ax = plt.subplots(figsize=(12, 6.75))
    bars = ax.barh(names, values, color=colors, edgecolor='white', linewidth=0.5)

    for bar, value in zip(bars, values):
        ax.text(value + 0.005, bar.get_y() + bar.get_height()/2,
                f"{value:.3f}", va='center', fontsize=11)

    ax.set_xlabel("nDCG@10", fontsize=13)
    ax.set_xlim(0, 0.95)
    ax.set_title("Retrieval Performance on Test Set",
                 fontsize=14, pad=15)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / "01_main_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")

def plot_capacity_story():
    """Dumbbell chart: baseline vs. fine-tuned for MiniLM and BGE."""
    models    = ["MiniLM\n(22M params)", "BGE-Large\n(335M params)"]
    baseline  = [RESULTS["MiniLM"]["ndcg"],    RESULTS["BGE"]["ndcg"]]
    finetuned = [RESULTS["MiniLM-FT"]["ndcg"], RESULTS["BGE-FT"]["ndcg"]]

    fig, ax = plt.subplots(figsize=(11, 5))
    y_positions = list(range(len(models)))

    for i, (b, f) in enumerate(zip(baseline, finetuned)):
        ax.plot([b, f], [i, i], color='gray', linewidth=2.5, zorder=1)
        mid = (b + f) / 2
        delta = f - b
        ax.text(mid, i + 0.15, f"+{delta:.3f}", ha='center', va='bottom',
                fontsize=11, color='gray', fontweight='bold')

    ax.scatter(baseline, y_positions, s=220, color=COLOR_DEFAULT,
               zorder=2, label="Baseline", edgecolor='white', linewidth=1.5)
    ax.scatter(finetuned, y_positions, s=220, color=COLOR_HIGHLIGHT,
               zorder=3, label="Fine-tuned", edgecolor='white', linewidth=1.5)

    for i, (b, f) in enumerate(zip(baseline, finetuned)):
        ax.text(b - 0.012, i, f"{b:.3f}", ha='right', va='center', fontsize=10)
        ax.text(f + 0.012, i, f"{f:.3f}", ha='left',  va='center', fontsize=10)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(models, fontsize=12)
    ax.set_xlabel("nDCG@10", fontsize=13)
    ax.set_xlim(0, 0.95)
    ax.set_title("Fine-tuning Capacity: same data, same method, different ceilings",
                 fontsize=14, pad=15)
    ax.legend(loc='lower right', fontsize=11, frameon=False)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / "02_capacity_story.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")

def plot_training_curves_zoomed():
    """Line chart: nDCG@10 on dev_monitor per epoch, BGE and MiniLM.
    Y-axis auto-zoomed for maximum trend readability."""
    epochs       = [0, 1, 2, 3, 4, 5]
    bge_curve    = [0.7046, 0.7163, 0.7373, 0.7595, 0.7663, 0.7643]
    minilm_curve = [0.6323, 0.6516, 0.6630, 0.6758, 0.6744, 0.6738]

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(epochs, bge_curve, marker='o', markersize=8, linewidth=2.5,
            color=COLOR_HIGHLIGHT, label="BGE-Large (335M params)")
    ax.plot(epochs, minilm_curve, marker='o', markersize=8, linewidth=2.5,
            color=COLOR_DEFAULT, label="MiniLM (22M params)")

    bge_best_idx, minilm_best_idx = 4, 3
    ax.scatter([bge_best_idx], [bge_curve[bge_best_idx]],
               marker='*', s=400, color=COLOR_HIGHLIGHT,
               edgecolor='black', linewidth=1.2, zorder=5)
    ax.scatter([minilm_best_idx], [minilm_curve[minilm_best_idx]],
               marker='*', s=400, color=COLOR_DEFAULT,
               edgecolor='black', linewidth=1.2, zorder=5)

    bge_delta    = bge_curve[bge_best_idx]       - bge_curve[0]
    minilm_delta = minilm_curve[minilm_best_idx] - minilm_curve[0]
    ax.text(5.1, bge_curve[bge_best_idx],
            f"Δ = +{bge_delta:.3f}",
            color=COLOR_HIGHLIGHT, fontsize=11, fontweight='bold', va='center')
    ax.text(5.1, minilm_curve[minilm_best_idx],
            f"Δ = +{minilm_delta:.3f}",
            color=COLOR_DEFAULT, fontsize=11, fontweight='bold', va='center')

    ax.set_xlabel("Epoch (0 = pretrained baseline)", fontsize=13)
    ax.set_ylabel("nDCG@10 on dev_monitor (150 claims)", fontsize=13)
    ax.set_xticks(epochs)
    ax.set_xlim(-0.3, 6.0)
    ax.set_title("Both models steadily improved during training",
                 fontsize=14, pad=15)
    ax.legend(loc='lower right', fontsize=11, frameon=False)
    ax.grid(linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / "03_training_curves_zoomed.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def plot_training_curves_full_scale():
    """Same line chart as above, but Y-axis fixed to 0-1 (scientifically stricter).
    Trends look flatter, but the scale is honest."""
    epochs       = [0, 1, 2, 3, 4, 5]
    bge_curve    = [0.7046, 0.7163, 0.7373, 0.7595, 0.7663, 0.7643]
    minilm_curve = [0.6323, 0.6516, 0.6630, 0.6758, 0.6744, 0.6738]

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(epochs, bge_curve, marker='o', markersize=8, linewidth=2.5,
            color=COLOR_HIGHLIGHT, label="BGE-Large (335M params)")
    ax.plot(epochs, minilm_curve, marker='o', markersize=8, linewidth=2.5,
            color=COLOR_DEFAULT, label="MiniLM (22M params)")

    bge_best_idx, minilm_best_idx = 4, 3
    ax.scatter([bge_best_idx], [bge_curve[bge_best_idx]],
               marker='*', s=400, color=COLOR_HIGHLIGHT,
               edgecolor='black', linewidth=1.2, zorder=5)
    ax.scatter([minilm_best_idx], [minilm_curve[minilm_best_idx]],
               marker='*', s=400, color=COLOR_DEFAULT,
               edgecolor='black', linewidth=1.2, zorder=5)

    bge_delta    = bge_curve[bge_best_idx]       - bge_curve[0]
    minilm_delta = minilm_curve[minilm_best_idx] - minilm_curve[0]
    ax.text(5.1, bge_curve[bge_best_idx],
            f"Δ = +{bge_delta:.3f}",
            color=COLOR_HIGHLIGHT, fontsize=11, fontweight='bold', va='center')
    ax.text(5.1, minilm_curve[minilm_best_idx],
            f"Δ = +{minilm_delta:.3f}",
            color=COLOR_DEFAULT, fontsize=11, fontweight='bold', va='center')

    ax.set_xlabel("Epoch (0 = pretrained baseline)", fontsize=13)
    ax.set_ylabel("nDCG@10 on dev_monitor (150 claims)", fontsize=13)
    ax.set_xticks(epochs)
    ax.set_xlim(-0.3, 6.0)
    ax.set_ylim(0, 1)
    ax.set_title("Both models steadily improved during training",
                 fontsize=14, pad=15)
    ax.legend(loc='lower right', fontsize=11, frameon=False)
    ax.grid(linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / "03_training_curves_full_scale.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def plot_reranking_effect():
    """Slope chart: reranking effect per retriever.
    Shows: weak retrievers benefit a lot, BGE-FT is hurt by the cross-encoder."""
    methods = {
        "MiniLM":    {"without": 0.655, "with": 0.775},
        "MiniLM-FT": {"without": 0.725, "with": 0.791},
        "BGE":       {"without": 0.786, "with": 0.795},
        "BGE-FT":    {"without": 0.830, "with": 0.798},
    }

    fig, ax = plt.subplots(figsize=(11, 6.5))
    x_positions = [0, 1]
    x_labels = ["without\nReranker", "with mxbai\nReranker"]

    for method, vals in methods.items():
        is_negative = method == "BGE-FT"
        color = COLOR_WARNING if is_negative else COLOR_DEFAULT
        linewidth = 3.0 if is_negative else 2.0

        ax.plot(x_positions, [vals["without"], vals["with"]],
                marker='o', markersize=10, linewidth=linewidth, color=color)

        delta = vals["with"] - vals["without"]
        sign = "+" if delta >= 0 else ""

        ax.text(1.05, vals["with"], f"{method} ({sign}{delta:.3f})",
                va='center', fontsize=11, color=color,
                fontweight='bold' if is_negative else 'normal')

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_xlim(-0.1, 1.55)
    ax.set_ylabel("nDCG@10 on dev_test", fontsize=13)
    ax.set_title("Reranking helps weak retrievers — but hurts the strongest",
                 fontsize=14, pad=15)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / "04_reranking_effect.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")

def plot_nli_confusion_matrices():
    """Side-by-side confusion matrices for the NLI verdict step, gold-evidence
    pass on the 300 dev claims (results/eval_nli.log, results/eval_nli_finetuned.log).

    Left: zero-shot roberta-large-nli. Right: the SciFact fine-tuned classifier.
    The zero-shot model dumps most SUPPORT and CONTRADICT claims into NEI;
    fine-tuning drains that over-predicted column.
    """
    labels = ["SUPPORT", "CONTRADICT", "NEI"]

    # rows = gold verdict, columns = predicted verdict
    zero_shot = np.array([
        [53,  3, 68],
        [ 7, 16, 41],
        [ 6,  7, 99],
    ])
    finetuned = np.array([
        [100, 12, 12],
        [ 16, 40,  8],
        [ 11,  6, 95],
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2))
    fig.subplots_adjust(wspace=0.30, top=0.74, bottom=0.12)
    panels = [
        (axes[0], zero_shot,
         "Zero-shot (roberta-large-nli)\nacc 0.560   macro-F1 0.511"),
        (axes[1], finetuned,
         "Fine-tuned on SciFact (nli-roberta-base)\nacc 0.783   macro-F1 0.763"),
    ]

    for ax, matrix, title in panels:
        row_sums = matrix.sum(axis=1, keepdims=True)
        recall = matrix / row_sums

        ax.imshow(recall, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")

        for i in range(3):
            for j in range(3):
                share = recall[i, j]
                txt_color = "white" if share > 0.55 else "#1a1a1a"
                ax.text(j, i, f"{matrix[i, j]}\n{share:.0%}", ha="center", va="center",
                        fontsize=13, fontweight="bold", color=txt_color, linespacing=1.5)

        # outline the correct (diagonal) cells
        for k in range(3):
            ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False,
                                       edgecolor=COLOR_HIGHLIGHT, linewidth=3))

        wrong_to_nei = int(matrix[0, 2] + matrix[1, 2])
        ax.set_xticks(range(3), labels, fontsize=11)
        ax.set_yticks(range(3), labels, fontsize=11, rotation=90, va="center")
        ax.xaxis.set_ticks_position("top")
        ax.set_xlabel("predicted", fontsize=13)
        ax.xaxis.set_label_position("top")
        ax.set_ylabel("gold", fontsize=13)
        ax.set_title(title, fontsize=12, pad=26)
        ax.text(0.5, -0.14,
                f"SUPPORT / CONTRADICT misread as NEI:  {wrong_to_nei}",
                transform=ax.transAxes, ha="center", fontsize=11,
                color=COLOR_WARNING, fontweight="bold")
        ax.set_xticks(np.arange(-0.5, 3), minor=True)
        ax.set_yticks(np.arange(-0.5, 3), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)

    fig.suptitle("Fine-tuning drains the over-predicted NEI column  ·  gold evidence, 300 dev claims",
                 fontsize=13, y=0.98)
    out = OUTPUT_DIR / "05_nli_confusion_matrices.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_nli_metrics_comparison():
    """Grouped bars: accuracy and macro-F1 for the zero-shot vs. fine-tuned NLI
    model, each scored on retrieved evidence (BGE-FT top-1) and on gold evidence.

    Nuance: the gold-minus-retrieved gap *widens* after fine-tuning. Once the
    classifier is strong, retrieval mistakes become the visible bottleneck.
    """
    # (label, retrieved evidence, gold evidence)  -- results/eval_nli*.log
    rows = [
        ("Accuracy\nzero-shot",  0.543, 0.560),
        ("Accuracy\nfine-tuned", 0.710, 0.783),
        ("Macro-F1\nzero-shot",  0.489, 0.511),
        ("Macro-F1\nfine-tuned", 0.697, 0.763),
    ]
    labels    = [r[0] for r in rows]
    retrieved = [r[1] for r in rows]
    gold      = [r[2] for r in rows]

    x = np.arange(len(rows))
    bar_w = 0.38

    fig, ax = plt.subplots(figsize=(12, 6.75))
    b1 = ax.bar(x - bar_w / 2, retrieved, bar_w, color=COLOR_DEFAULT,
                edgecolor="white", linewidth=0.5, label="retrieved (BGE-FT top-1)")
    b2 = ax.bar(x + bar_w / 2, gold, bar_w, color=COLOR_HIGHLIGHT,
                edgecolor="white", linewidth=0.5, label="gold evidence")

    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{bar.get_height():.3f}", ha="center", fontsize=11)

    for xi, (_, r, g) in enumerate(rows):
        top = max(r, g) + 0.07
        ax.annotate("", xy=(xi + bar_w / 2, top), xytext=(xi - bar_w / 2, top),
                    arrowprops=dict(arrowstyle="<->", color="gray", lw=1.2))
        ax.text(xi, top + 0.02, f"Δ +{g - r:.3f}", ha="center", fontsize=10,
                color="gray", fontweight="bold")

    ax.set_xticks(x, labels, fontsize=11)
    ax.set_ylabel("score on the 300 dev claims", fontsize=13)
    ax.set_ylim(0, 0.95)
    ax.set_title("NLI verdict: fine-tuning lifts every metric — and widens the retrieval gap",
                 fontsize=14, pad=15)
    ax.legend(loc="upper left", fontsize=11, frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / "06_nli_metrics_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_arctic_side_experiment():
    """Side experiment, not part of the final pipeline: Snowflake
    Arctic-embed-l-v2.0 vs. BGE-Large, pretrained vs. fine-tuned,
    nDCG@10 on the dev_monitor split (logs/eval_arctic.log,
    logs/finetune_arctic.log).

    Arctic-FT edges out BGE-FT here (0.779 vs 0.766) — but dev_monitor was the
    fine-tuning monitor split, so both fine-tuned numbers are optimistic, and
    only BGE-FT was re-indexed and confirmed on the held-out dev-test split
    (0.830, results/eval_dev_test.log). The Arctic result was never validated
    end to end, so BGE-FT stayed the chosen model.
    """
    models = ["Arctic-embed-l-v2.0", "BGE-Large"]
    pretrained = [0.691, 0.705]
    finetuned = [0.779, 0.766]
    validated_bge_ft_devtest = 0.830

    x = np.arange(len(models))
    bar_w = 0.34

    fig, ax = plt.subplots(figsize=(11, 6.5))

    b1 = ax.bar(x - bar_w / 2, pretrained, bar_w, color=COLOR_DEFAULT,
                edgecolor="white", linewidth=0.5, label="pretrained")
    b2 = ax.bar(x + bar_w / 2, finetuned, bar_w, color=COLOR_HIGHLIGHT,
                edgecolor="white", linewidth=0.5, label="fine-tuned on SciFact")

    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{bar.get_height():.3f}", ha="center", fontsize=11)

    # the Arctic fine-tuned score was never re-scored on truly held-out data
    ax.text(x[0] + bar_w / 2, finetuned[0] / 2,
            "never re-scored\non held-out data", ha="center", va="center",
            fontsize=10, color="white", fontweight="bold")
    ax.text(x[1] + bar_w / 2, finetuned[1] / 2,
            "carried forward\n(see dashed line)", ha="center", va="center",
            fontsize=10, color="white", fontweight="bold")

    ax.axhline(validated_bge_ft_devtest, color=COLOR_WARNING, linestyle="--", linewidth=1.8)
    ax.text(len(models) - 0.5, validated_bge_ft_devtest + 0.008,
            f"BGE-FT on held-out dev-test: {validated_bge_ft_devtest:.3f}  (validated)",
            ha="right", fontsize=10, color=COLOR_WARNING, fontweight="bold")

    ax.set_xticks(x, models, fontsize=12)
    ax.set_ylabel("nDCG@10 on dev_monitor (150 claims)", fontsize=13)
    ax.set_ylim(0, 0.95)
    ax.set_title("Arctic-embed: a promising side experiment I didn't validate",
                 fontsize=14, pad=15)
    ax.legend(loc="upper left", fontsize=11, frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.text(0.5, -0.16,
            "dev_monitor was the fine-tuning monitor split — both fine-tuned bars are mild upper bounds.\n"
            "Only BGE-FT was carried forward and re-scored on the held-out dev-test split.",
            transform=ax.transAxes, ha="center", fontsize=9.5, color="#555555")

    plt.tight_layout()
    out = OUTPUT_DIR / "07_arctic_side_experiment.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    plot_main_comparison()
    plot_capacity_story()
    plot_training_curves_zoomed()
    plot_training_curves_full_scale()
    plot_reranking_effect()
    plot_nli_confusion_matrices()
    plot_nli_metrics_comparison()
    plot_arctic_side_experiment()
