#!/usr/bin/env python3
"""
Default Griffin trainer entrypoint.

This now points to the legacy Griffin implementation, which is the current
best-performing Griffin backbone in this codebase.

The more complex experimental Griffin/RecurrentGemma-style path has been
preserved in `train_griffin_modern.py` for reference, but it is no longer the
default training path.
"""

from train_griffin_old import *  # noqa: F401,F403
from train_griffin_old import main


if __name__ == "__main__":
    main()
