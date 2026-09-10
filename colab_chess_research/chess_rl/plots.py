"""Portable matplotlib figures from actual saved metrics only."""

import csv
from pathlib import Path
import matplotlib.pyplot as plt
from .reproducibility import read_json

COLORS = {"policy": "#2274a5", "value": "#bd552b", "reference": "#444444", "score": "#287b64"}


def setup():
    plt.rcParams.update(
        {
            "figure.figsize": (9, 4.5),
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "figure.dpi": 110,
        }
    )


def save(fig, root, run_id, name):
    directory = Path(root) / "results" / run_id / "plots"
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / (name + ".png"), dpi=160, bbox_inches="tight")
    fig.savefig(directory / (name + ".pdf"), bbox_inches="tight")
    return fig


def plot_supervised(root, run_id):
    setup()
    path = Path(root) / "logs" / run_id / "supervised.csv"
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("No completed supervised epochs to plot")
    epochs = [int(row["epoch"]) for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for key, label, color in (
        ("policy_loss", "Validation policy CE", COLORS["policy"]),
        ("value_loss", "Validation value Huber", COLORS["value"]),
    ):
        axes[0, 0].plot(
            epochs, [float(row[key]) for row in rows], label=label, color=color, marker="o"
        )
    axes[0, 0].legend()
    axes[0, 0].set(title="Validation losses", xlabel="Completed epoch", ylabel="Mean loss")
    for key, style in (("top1", "-"), ("top3", "--")):
        axes[0, 1].plot(
            epochs, [100 * float(row[key]) for row in rows], label=key, linestyle=style, marker="o"
        )
    axes[0, 1].set(
        title="Legal-move target accuracy",
        xlabel="Completed epoch",
        ylabel="Accuracy (%)",
        ylim=(0, 100),
    )
    axes[0, 1].legend()
    axes[1, 0].plot(
        epochs, [float(row["value_mae"]) for row in rows], color=COLORS["value"], marker="o"
    )
    axes[1, 0].set(
        title="Value target error", xlabel="Completed epoch", ylabel="Mean absolute error"
    )
    axes[1, 1].plot(
        epochs,
        [float(row["epoch_seconds"]) / 60 for row in rows],
        marker="o",
        color=COLORS["reference"],
    )
    axes[1, 1].set(title="Training and validation time", xlabel="Completed epoch", ylabel="Minutes")
    fig.suptitle(f"Supervised run: {run_id}")
    return save(fig, root, run_id, "supervised")


def plot_matches(root, run_id, summaries, name="match_comparison"):
    setup()
    fig, ax = plt.subplots(layout="constrained")
    available = [(label, value) for label, value in summaries.items() if value["score"] is not None]
    if not available:
        raise ValueError("No scored matches to plot")
    for index, (label, result) in enumerate(available):
        score = result["score"] * 100
        low, high = result["interval95"]
        if low is None:
            ax.plot(score, index, "o", color=COLORS["score"])
        else:
            ax.hlines(index, low * 100, high * 100, color=COLORS["score"])
            ax.plot(score, index, "o", color=COLORS["score"])
    ax.set_yticks(
        range(len(available)), [f"{label} (n={r['scored_games']})" for label, r in available]
    )
    ax.axvline(50, color=COLORS["reference"], linestyle="--")
    ax.set(
        xlim=(0, 100),
        xlabel="Score (%) with paired 95% bootstrap interval",
        title="Candidate score by opponent or ablation",
    )
    return save(fig, root, run_id, name)


def plot_league(root, run_id):
    setup()
    state = read_json(Path(root) / "checkpoints/self_play" / run_id / "league.json")
    entries = state["iterations"]
    if not entries:
        raise ValueError("No completed league iterations")
    fig, ax = plt.subplots(layout="constrained")
    for entry in entries:
        result = entry["summary"]
        if result["score"] is None:
            continue
        ax.plot(
            entry["iteration"],
            result["score"] * 100,
            "o",
            color=COLORS["score"],
            markerfacecolor=COLORS["score"] if entry["promoted"] else "none",
        )
    ax.axhline(50, color=COLORS["reference"], linestyle="--")
    ax.set(
        xlabel="Completed iteration",
        ylabel="Development score against incumbent (%)",
        ylim=(0, 100),
        title="League candidates (filled marker = promoted)",
    )
    return save(fig, root, run_id, "league")
