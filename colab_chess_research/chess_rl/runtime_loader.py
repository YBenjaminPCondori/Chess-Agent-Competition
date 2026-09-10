"""Minimal runtime reconstruction; imported by the final candidate, not by training."""

from pathlib import Path
import torch
from .model import ChessPolicyValueNetSmall


class OnnxModel(torch.nn.Module):
    def __init__(self, path):
        super().__init__()
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def forward(self, board):
        logits, values = self.session.run(None, {"board": board.detach().cpu().numpy()})
        return torch.from_numpy(logits), torch.from_numpy(values)


def reconstruct(directory, config):
    directory = Path(directory)
    if config["format"] == "onnx":
        return OnnxModel(directory / "weights/best_model.onnx").eval()
    model = ChessPolicyValueNetSmall(**config["architecture"]).cpu().eval()
    if config.get("quantization") == "dynamic_int8":
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    state = torch.load(
        directory / "weights/best_model_cpu.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(state, strict=True)
    return model.eval()
