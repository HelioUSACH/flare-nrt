"""Ensemble of 5 ResNet-18 (8 input channels, one logit); output = mean of the
members' sigmoids. The weights file is verified against its SHA-256 before
loading."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn as nn
import torchvision


class WeightsError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_member() -> nn.Module:
    m = torchvision.models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m


def clean_keys(sd: dict) -> dict:
    """Training checkpoints wrap the state dict and name the head
    ``head.1`` (Sequential(Dropout, Linear)); map to torchvision names."""
    for wrap in ("model_state", "state_dict", "model"):
        if isinstance(sd, dict) and isinstance(sd.get(wrap), dict):
            sd = sd[wrap]
    out = {}
    for k, v in sd.items():
        for pre in ("module.", "model.", "backbone.", "net."):
            if k.startswith(pre):
                k = k[len(pre):]
        if k.startswith("head.1."):
            k = "fc." + k[len("head.1."):]
        out[k] = v
    return out


class Ensemble:
    def __init__(self, members: list[nn.Module], meta: dict, device: str = "cpu"):
        self.members = members
        self.meta = meta
        self.device = device

    @classmethod
    def load(cls, path: Path, expected_sha256: str = "", device: str | None = None) -> Ensemble:
        path = Path(path)
        if not path.exists():
            raise WeightsError(f"weights not found: {path}")
        digest = sha256(path)
        if expected_sha256 and digest != expected_sha256.lower():
            raise WeightsError(f"SHA-256 mismatch for {path.name}: {digest}")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            # Older checkpoints carry numpy scalars in their metadata. Only
            # reached after the SHA-256 check above has passed.
            if not expected_sha256:
                raise WeightsError("checkpoint needs weights_only=False; set FLARE_WEIGHTS_SHA256") from None
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        members = []
        for i, entry in enumerate(ckpt["ensemble_members"]):
            net = build_member()
            try:
                net.load_state_dict(clean_keys(entry), strict=True)
            except RuntimeError as e:
                raise WeightsError(f"member {i}: keys don't match ResNet-18/8ch") from e
            members.append(net.eval().to(device))
        meta = {"sha256": digest, "n_members": len(members),
                "seeds": [e.get("seed") for e in ckpt["ensemble_members"] if isinstance(e, dict)]}
        return cls(members, meta, device)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[float, list[float]]:
        """x: (8, 512, 512). Returns (mean probability, per-member scores), fp32."""
        xb = x.float().unsqueeze(0).to(self.device)
        scores = [float(torch.sigmoid(m(xb)).squeeze()) for m in self.members]
        return float(sum(scores) / len(scores)), scores
