#!/usr/bin/env python3
"""Synthetic smoke test (same as examples/train_synthetic.py)."""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "examples" / "train_synthetic.py"), run_name="__main__")
