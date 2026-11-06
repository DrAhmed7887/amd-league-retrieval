"""Static checks for the hard gates that score zero regardless of the model."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _from_lines(path: Path):
    return [line.split()[1] for line in path.read_text().splitlines() if line.strip().upper().startswith("FROM ")]


def test_final_stage_uses_the_mandated_base():
    starter = ROOT / "kit" / "starter" / "Dockerfile"
    if not starter.exists():
        pytest.skip("starter kit not unpacked")
    assert _from_lines(ROOT / "Dockerfile")[-1] == _from_lines(starter)[-1]


def test_requirements_are_pinned_and_never_touch_torch():
    lines = [l.strip() for l in (ROOT / "requirements.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
    assert lines and all("==" in l for l in lines)
    assert not any(re.match(r"(torch|torchvision|torchaudio|triton)\b", l) for l in lines)


def test_dockerfile_guards_rocm_torch_and_ships_weights_offline():
    text = (ROOT / "Dockerfile").read_text()
    assert "-c /tmp/torch-pins.txt" in text and "'+rocm' in v" in text
    assert re.search(r"MODEL_REVISION=[0-9a-f]{40}", text)
    assert "HF_HUB_OFFLINE=1" in text and 'CMD ["python3", "/app/server.py"]' in text
    assert "--squash" not in text


def test_build_context_leaves_out_the_kit_and_tests():
    ignored = (ROOT / ".dockerignore").read_text().split()
    assert {"kit", "tests", ".venv", ".git"} <= set(ignored)
