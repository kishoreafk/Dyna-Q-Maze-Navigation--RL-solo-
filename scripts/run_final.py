#!/usr/bin/env python3
"""Run the full final protocol (30 map/seed pairs, 4 scenarios, 3 slips).

This is the same pipeline as run_pilot.py with configs/final.yaml; it was
NOT executed inside the delivery sandbox (runtime), but is fully supported.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from run_pilot import main  # noqa: E402,F401

if __name__ == "__main__":
    sys.argv += ["--config", "configs/final.yaml"] if "--config" not in " ".join(sys.argv) else []
    main()
