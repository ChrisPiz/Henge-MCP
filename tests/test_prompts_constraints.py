"""Each frame prompt must have a `## CONSTRAINTS` section with MUST and CANNOT bullets."""
from pathlib import Path

import pytest


_PROMPTS_DIR = Path(__file__).parent.parent / "henge" / "prompts"

_FRAMES_TO_FILES = {
    "empirical":         "01-empirical.md",
    "historical":        "02-historical.md",
    "first-principles":  "03-first-principles.md",
    "analogical":        "04-analogical.md",
    "systemic":          "05-systemic.md",
    "ethical":           "06-ethical.md",
    "soft-contrarian":   "07-soft-contrarian.md",
    "radical-optimist":  "08-radical-optimist.md",
    "pre-mortem":        "09-pre-mortem.md",
}


@pytest.mark.parametrize("frame,filename", list(_FRAMES_TO_FILES.items()))
def test_prompt_has_constraints_section(frame, filename):
    text = (_PROMPTS_DIR / filename).read_text(encoding="utf-8")
    assert "## CONSTRAINTS" in text, f"{filename} missing ## CONSTRAINTS"
    assert "You MUST" in text or "MUST:" in text, f"{filename} missing MUST list"
    assert "You CANNOT" in text or "CANNOT:" in text, f"{filename} missing CANNOT list"


@pytest.mark.parametrize("frame,filename", list(_FRAMES_TO_FILES.items()))
def test_prompt_has_output_format(frame, filename):
    text = (_PROMPTS_DIR / filename).read_text(encoding="utf-8")
    assert "Output format" in text, f"{filename} missing Output format spec"
