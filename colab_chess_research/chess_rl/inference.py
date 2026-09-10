"""Shared inference adapter. Exported callers supply CPU tensors and JSON config."""

import time
import torch
from .board_encoding import encode_board
from .action_encoding import encode_move


class NeuralEvaluator:
    def __init__(self, model, device="cpu", policy_ordering=True, value_mix=0.5):
        self.model = model.to(device).eval() if model is not None else None
        self.device = device
        self.policy_ordering = policy_ordering
        self.value_mix = value_mix
        self.latency_ms = 10.0
        self.stats = dict(
            inference_calls=0,
            inference_ms=0.0,
            inference_max_ms=0.0,
            model_errors=0,
            budget_fallbacks=0,
            missing_model=0,
        )
        self.last_error = None

    def predict(self, board, deadline):
        if self.model is None:
            self.stats["missing_model"] += 1
            return None
        if deadline.remaining_ms() < max(5.0, self.latency_ms * 1.5 + 2):
            self.stats["budget_fallbacks"] += 1
            return None
        deadline.check()
        started = time.monotonic()
        try:
            with torch.inference_mode():
                logits, values = self.model(encode_board(board).unsqueeze(0).to(self.device))
                logits, value = logits[0].float().cpu(), float(values[0].float().item())
            if not torch.isfinite(logits).all() or not -1.00001 <= value <= 1.00001:
                raise ValueError("Nonfinite or out-of-range neural output")
        except Exception as error:
            self.stats["model_errors"] += 1
            self.last_error = f"{type(error).__name__}: {error}"
            self.model = None
            return None
        finally:
            elapsed = (time.monotonic() - started) * 1000
            self.latency_ms = max(elapsed, 0.9 * self.latency_ms)
            self.stats["inference_calls"] += 1
            self.stats["inference_ms"] += elapsed
            self.stats["inference_max_ms"] = max(self.stats["inference_max_ms"], elapsed)
        deadline.check()
        return logits, value

    @staticmethod
    def ranks(board, moves, logits):
        return {move: float(logits[encode_move(board, move)]) for move in moves}
