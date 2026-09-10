"""A resumable opponent league. Selection records point at immutable checkpoints."""

import random
from pathlib import Path
from .reproducibility import atomic_json, read_json, sha256


class League:
    def __init__(self, root, cfg, initial_checkpoint):
        self.root = Path(root)
        self.path = self.root / "checkpoints/self_play" / cfg["run_id"] / "league.json"
        self.cfg = cfg
        initial = str(Path(initial_checkpoint).resolve().relative_to(self.root.resolve()))
        if self.path.exists():
            self.state = read_json(self.path)
            if self.state["config"] != cfg or self.state["initial_sha256"] != sha256(
                initial_checkpoint
            ):
                raise ValueError("League initialization/config changed under this run_id")
        else:
            self.state = dict(
                config=cfg,
                initial_sha256=sha256(initial_checkpoint),
                champion=initial,
                history=[initial],
                completed_iterations=0,
                iterations=[],
            )
            self.save()

    def save(self):
        atomic_json(self.path, self.state)

    def opponent(self, seed):
        rng = random.Random(seed)
        history = self.state["history"]
        choices = [
            ("neural", self.state["champion"], 0.40),
            ("classical", "reference/classical_agent/agent.py", 0.10),
            ("baseline", "reference/starter/baselines/greedy/agent.py", 0.05),
            ("baseline", "reference/starter/baselines/minimax/agent.py", 0.05),
        ]
        if len(history) >= 2:
            choices.append(("neural", history[-2], 0.25))
        if len(history) >= 3:
            for older in history[:-2]:
                choices.append(("neural", older, 0.15 / len(history[:-2])))
        total = sum(c[2] for c in choices)
        normalized = [(kind, path, weight / total) for kind, path, weight in choices]
        choice = rng.choices(normalized, weights=[c[2] for c in normalized], k=1)[0]
        return choice, normalized

    def finish_iteration(self, number, candidate, promoted, summary):
        relative = str(Path(candidate).resolve().relative_to(self.root.resolve()))
        if number != self.state["completed_iterations"] + 1:
            raise ValueError("League iteration out of order")
        if promoted:
            self.state["champion"] = relative
            self.state["history"].append(relative)
        self.state["iterations"].append(
            dict(iteration=number, candidate=relative, promoted=promoted, summary=summary)
        )
        self.state["completed_iterations"] = number
        self.save()

    @property
    def champion(self):
        return self.root / self.state["champion"]
