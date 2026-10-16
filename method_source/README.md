# Recorded implementation

These are the actual experiment entrypoints kept for method inspection. They reference the original experiment workspace and are not standalone run commands. The portable training objective is in `src/training_objective.py`; `scripts/train.py` uses it with the exported records and fixed schedule. The portable script runs 1,000 updates continuously; the recorded run paused at 200 and resumed the same optimizer and RNG. Fresh-run bitwise equivalence is not claimed.
