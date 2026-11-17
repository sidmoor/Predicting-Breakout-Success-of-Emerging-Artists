"""Fetch every MusicBrainz release from RELEASES_START to the panel end via LB fresh-releases.

Each call returns releases in the `days` (max 90) before `release_date`. Responses are cached in
data/raw/lb_api/, so reruns only fetch missing windows. Output: data/interim/releases_raw.parquet
(one row per release, deduplicated).

    python scripts/fetch_releases.py
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from release_impact.config import INTERIM, PANEL_END_EXCL, RAW, RELEASES_START, USER_AGENT  # noqa: E402

CACHE = RAW / "lb_api"
session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})
session.mount("https://", HTTPAdapter(max_retries=Retry(
    total=5, backoff_factor=5, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=None)))


def fresh_releases(release_date, days=90):
    """Releases dated in the `days` before release_date (inclusive), cached on disk."""
    suffix = "" if days == 90 else f"_{days}d"
    cache = CACHE / f"fresh_releases_{release_date:%Y-%m-%d}{suffix}.json"
    if not cache.exists():
        t0 = time.time()
        resp = session.get("https://api.listenbrainz.org/1/explore/fresh-releases/", timeout=300, params={
            "release_date": f"{release_date:%Y-%m-%d}", "days": days, "past": "true", "future": "false"})
        resp.raise_for_status()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(resp.text, encoding="utf-8")
        print(f"  fetched {cache.name} ({time.time() - t0:.0f}s)", flush=True)
    return json.loads(cache.read_text(encoding="utf-8"))["payload"]["releases"]


def main():
    # Quarterly 90-day windows (matches the existing cache), then fill any missing days with short calls
    rows = []
    for end in pd.date_range(RELEASES_START + pd.offsets.QuarterBegin(startingMonth=1), PANEL_END_EXCL + pd.offsets.QuarterBegin(startingMonth=1), freq="QS"):
        rows += fresh_releases(end)
    df = pd.DataFrame(rows)
    covered = set(pd.to_datetime(df["release_date"]).dt.normalize())
    wanted = pd.date_range(RELEASES_START, PANEL_END_EXCL, freq="D")
    missing = [d for d in wanted if d not in covered]
    print(f"{len(missing)} days with no releases after the quarterly calls; filling with 3-day calls")
    filled = set()
    for d in missing:
        if d in filled:
            continue
        end = d + pd.Timedelta(days=2)
        rows += fresh_releases(end, days=3)
        filled.update(pd.date_range(d, end))

    df = pd.DataFrame(rows).drop_duplicates("release_mbid")
    df["release_date"] = pd.to_datetime(df["release_date"])
    df = df[(df["release_date"] >= RELEASES_START) & (df["release_date"] < PANEL_END_EXCL)]
    df = df.drop(columns=["listen_count", "caa_id", "caa_release_mbid"], errors="ignore")
    df["release_tags"] = df["release_tags"].map(lambda t: "; ".join(t or []))
    out = INTERIM / "releases_raw.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Saved {out}: {len(df):,} releases, {df['release_group_mbid'].nunique():,} release groups, "
          f"{df['release_date'].min():%Y-%m-%d} to {df['release_date'].max():%Y-%m-%d}")


if __name__ == "__main__":
    main()
