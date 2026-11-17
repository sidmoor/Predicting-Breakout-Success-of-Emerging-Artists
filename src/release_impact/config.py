"""Shared paths and windows for the release impact pipeline."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"

USER_AGENT = "TeamTalentScouts/0.1 ( https://github.com/sidmoor/Predicting-Breakout-Success-of-Emerging-Artists )"

# Listening panel: Friday-start weeks (most releases come out on a Friday, so release day starts a week).
# 2021-01-01 is a Friday. The dump ends 2026-11-01, so the last complete week starts 2026-10-23.
PANEL_START = pd.Timestamp("2021-01-01")
PANEL_END_EXCL = pd.Timestamp("2026-10-30")

# Releases come from LB fresh-releases (MusicBrainz data). History from 2020 gives "days since last release".
RELEASES_START = pd.Timestamp("2020-01-01")

# Release windows (in weeks): 12 before, 12 after (targets use weeks 0-3 and 8-11 after release)
PRE_WEEKS = 12
POST_WEEKS = 12
