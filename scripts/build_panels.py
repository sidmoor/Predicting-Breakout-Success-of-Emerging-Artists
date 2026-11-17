"""Combine the per-file weekly scan into exact weekly panels (data/processed/).

  platform_week.parquet   week, active_users, listens
  artist_week.parquet     week, artist_mbid, listens, listeners, repeat_listeners (2+ listens that week),
                          new_listeners (first listen to the artist since the panel start, 2021-01-01)
  recording_week.parquet  week, recording_mbid, listens
  release_recordings.parquet  release_mbid, recording_mbid, listens  (which tracks were heard on which release)

A week's listens are spread over ~2 dump files, so counts are re-aggregated here. Artist rows are
processed in hash buckets to keep DuckDB's memory bounded.

    python scripts/build_panels.py
"""
import shutil
import sys
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from release_impact.config import INTERIM, PROCESSED  # noqa: E402
from release_impact.lb_dump import log  # noqa: E402

N_BUCKETS = 16


def main():
    scan_dir = sorted(INTERIM.glob("weekly_scan_*"))[-1]
    PROCESSED.mkdir(parents=True, exist_ok=True)
    tmp = INTERIM / "duckdb_tmp"
    tmp.mkdir(exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET temp_directory = '{tmp.as_posix()}'; SET memory_limit = '8GB'; SET preserve_insertion_order = false")
    src = lambda kind: f"read_parquet('{scan_dir.as_posix()}/*_{kind}.parquet')"  # noqa: E731
    log(f"Building panels from {scan_dir.name}")

    t0 = time.time()
    con.execute(f"""COPY (
        SELECT week, count(DISTINCT user_id) AS active_users, sum(listens) AS listens
        FROM {src('plat')} GROUP BY week ORDER BY week
    ) TO '{(PROCESSED / 'platform_week.parquet').as_posix()}' (FORMAT parquet, COMPRESSION zstd)""")
    log(f"  platform_week ({time.time() - t0:.0f}s)")

    t0 = time.time()
    con.execute(f"""COPY (
        SELECT week, recording_mbid, sum(listens) AS listens FROM {src('rec')} GROUP BY ALL
    ) TO '{(PROCESSED / 'recording_week.parquet').as_posix()}' (FORMAT parquet, COMPRESSION zstd)""")
    con.execute(f"""COPY (
        SELECT release_mbid, recording_mbid, sum(listens) AS listens FROM {src('rec')}
        WHERE release_mbid IS NOT NULL GROUP BY ALL
    ) TO '{(PROCESSED / 'release_recordings.parquet').as_posix()}' (FORMAT parquet, COMPRESSION zstd)""")
    log(f"  recording_week + release_recordings ({time.time() - t0:.0f}s)")

    bucket_dir = INTERIM / "artist_week_buckets"
    shutil.rmtree(bucket_dir, ignore_errors=True)  # always rebuild: the scan may have grown since last run
    bucket_dir.mkdir()
    for b in range(N_BUCKETS):
        out = bucket_dir / f"{b}.parquet"
        t0 = time.time()
        con.execute(f"""COPY (
            WITH ua AS (
                SELECT week, user_id, artist_mbid, sum(listens) AS n
                FROM {src('ua')} WHERE hash(artist_mbid) % {N_BUCKETS} = {b}
                GROUP BY ALL
            ),
            first_seen AS (SELECT user_id, artist_mbid, min(week) AS first_week FROM ua GROUP BY ALL)
            SELECT week, artist_mbid,
                   sum(n)::INTEGER                          AS listens,
                   count(*)::INTEGER                        AS listeners,
                   count(*) FILTER (n >= 2)::INTEGER        AS repeat_listeners,
                   count(*) FILTER (week = first_week)::INTEGER AS new_listeners
            FROM ua JOIN first_seen USING (user_id, artist_mbid)
            GROUP BY ALL
        ) TO '{out.with_suffix('.tmp').as_posix()}' (FORMAT parquet, COMPRESSION zstd)""")
        out.with_suffix(".tmp").replace(out)
        log(f"  artist_week bucket {b + 1}/{N_BUCKETS} ({time.time() - t0:.0f}s)")
    con.execute(f"""COPY (SELECT * FROM read_parquet('{bucket_dir.as_posix()}/*.parquet') ORDER BY artist_mbid, week)
                    TO '{(PROCESSED / 'artist_week.parquet').as_posix()}' (FORMAT parquet, COMPRESSION zstd)""")
    n = con.sql(f"SELECT count(*), count(DISTINCT artist_mbid) FROM '{(PROCESSED / 'artist_week.parquet').as_posix()}'").fetchone()
    log(f"artist_week: {n[0]:,} rows, {n[1]:,} artists")


if __name__ == "__main__":
    main()
