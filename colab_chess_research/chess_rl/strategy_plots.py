"""Notebook tables and saved plots derived only from recorded research results."""

import csv
from pathlib import Path
import matplotlib.pyplot as plt


def dataset_summary(root, run_id, manifest):
    from IPython.display import Markdown, display

    rows = [dict(theme=theme, **counts) for theme, counts in sorted(manifest["themes"].items())]
    print("Label coverage:", manifest["label_coverage"])
    print("Reserved positions excluded:", manifest["reserved_positions_dropped"])
    print("Duplicate/cross-split rows excluded:", manifest["duplicate_or_cross_split_rows_dropped"])
    directory = Path(root) / "data/strategy" / manifest.get("dataset_id", run_id)
    with (directory / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["theme", "train", "validation", "test"])
        writer.writeheader()
        writer.writerows(rows)
    display(
        Markdown(
            "| Theme | Train | Validation | Test |\n| --- | ---: | ---: | ---: |\n"
            + "\n".join(
                f"| {r['theme']} | {r['train']} | {r['validation']} | {r['test']} |" for r in rows
            )
        )
    )
    fig, axis = plt.subplots(figsize=(10, 6))
    left = [0] * len(rows)
    for split in ("train", "validation", "test"):
        values = [r[split] for r in rows]
        axis.barh([r["theme"] for r in rows], values, left=left, label=split)
        left = [a + b for a, b in zip(left, values)]
    axis.set_xlabel("Validated positions")
    axis.legend()
    fig.tight_layout()
    fig.savefig(directory / "split_counts.png", dpi=160)
    return fig


def evaluation_summary(root, run_id, record):
    from IPython.display import Markdown, display

    rows = record["summaries"]

    def number(value):
        return "n/a" if value is None else f"{value:.3f}"

    display(
        Markdown(
            "| Candidate | Theme | N | Search | Policy top-1 | Policy top-3 | Value MAE | Legal | Pass/Fail/NA | Status |\n"
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n"
            + "\n".join(
                f"| {r['candidate']} | {r['theme']} | {r['positions']} | {number(r['accuracy'])} | "
                f"{number(r['policy_top1_agreement'])} | {number(r['policy_top3_agreement'])} | "
                f"{number(r['value_mae'])} | {number(r['legal_move_rate'])} | "
                f"{r['counts']['pass']}/{r['counts']['fail']}/{r['counts']['not_assessable']} | {r['status']} |"
                for r in rows
            )
        )
    )
    labelled = [r for r in rows if r["accuracy"] is not None]
    fig, axis = plt.subplots(figsize=(10, max(3, len(labelled) * 0.3)))
    if labelled:
        axis.barh(
            [r["candidate"] + ": " + r["theme"] for r in labelled],
            [r["accuracy"] for r in labelled],
        )
        axis.set_xlim(0, 1)
    else:
        axis.text(0.5, 0.5, "No move-labelled positions", ha="center")
    axis.set_xlabel("Search move agreement")
    fig.tight_layout()
    directory = Path(root) / "results/strategy_evaluation" / run_id / record["settings"]["split"]
    fig.savefig(directory / "move_agreement.png", dpi=160)
    return fig
