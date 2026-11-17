"""Scan the ListenBrainz Spark dump (2021 onward) into compact per-file weekly aggregates.

For every dump file in the panel window, writes three Parquet files to data/interim/weekly_scan_<dump>/:
  {file_no}_ua.parquet    (week, user_id, artist_mbid, listens)       - per user, so listeners and
                                                                        first-time listeners can be counted
  {file_no}_rec.parquet   (week, recording_mbid, release_mbid, listens) - track-level listen counts
  {file_no}_plat.parquet  (week, user_id, listens)                     - platform activity (unfiltered)

Artists with < 3 listeners and recordings with < 3 listens within a file are dropped (see MIN_IN_FILE).

Weeks start on Friday (see config). A week spans ~2 files, so these are partial counts that
build_panels.py combines. Files already written are skipped, so the scan resumes after a crash.

    python scripts/scan_dump_weekly.py [--workers 4] [--limit N]
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from release_impact.config import INTERIM, PANEL_END_EXCL, PANEL_START  # noqa: E402
from release_impact.lb_dump import first_file_on_or_after, latest_dump, log, open_member, parquet_index  # noqa: E402

COLS = ["listened_at", "user_id", "artist_credit_mbids", "recording_mbid", "release_mbid"]
MIN_IN_FILE = 3
WEEK = "CAST(date_trunc('week', listened_at - INTERVAL 4 DAY) + INTERVAL 4 DAY AS DATE)"  # Friday-start week


def scan_file(url, row, out_dir):
    file_no = int(row["file_no"])
    paths = {k: out_dir / f"{file_no}_{k}.parquet" for k in ("ua", "rec", "plat")}
    if all(p.exists() for p in paths.values()):
        return file_no, 0.0
    t0 = time.time()
    for attempt in range(6):  # the dump server throttles with 503s: back off and retry the whole file
        try:
            listens = pq.ParquetFile(open_member(url, row)).read(columns=COLS)
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 5:
                raise
            log(f"  file {file_no}: {type(e).__name__}, retrying in {60 * (attempt + 1)}s")
            time.sleep(60 * (attempt + 1))
    con = duckdb.connect()
    try:
        con.register("listens", listens)
        con.execute(f"""CREATE VIEW l AS SELECT *, {WEEK} AS week FROM listens
                        WHERE listened_at >= TIMESTAMP '{PANEL_START}' AND listened_at < TIMESTAMP '{PANEL_END_EXCL}'""")
        # Long-tail filter (keeps disk at ~6 MB/file instead of ~31 MB): drop artists with < MIN_IN_FILE
        # listeners and recordings with < MIN_IN_FILE listens within this file. Sorting before writing
        # lets zstd compress the repeated keys. Counts for very small artists are therefore lower bounds;
        # release eligibility needs >= 20 weekly listeners, so eligible artists are barely affected.
        queries = {
            "ua": f"""WITH ua AS (
                         SELECT week, user_id, artist_mbid, count(*) AS listens
                         FROM (SELECT week, user_id, unnest(artist_credit_mbids) AS artist_mbid FROM l
                               WHERE len(artist_credit_mbids) > 0)
                         GROUP BY ALL)
                      SELECT week, user_id, artist_mbid, listens::SMALLINT AS listens FROM ua
                      WHERE artist_mbid IN (SELECT artist_mbid FROM ua GROUP BY 1 HAVING count(*) >= {MIN_IN_FILE})
                      ORDER BY artist_mbid, week, user_id""",
            "rec": f"""WITH rec AS (
                          SELECT week, recording_mbid, release_mbid, count(*) AS listens
                          FROM l WHERE recording_mbid IS NOT NULL GROUP BY ALL)
                       SELECT week, recording_mbid, release_mbid, listens::INTEGER AS listens FROM rec
                       WHERE recording_mbid IN (SELECT recording_mbid FROM rec GROUP BY 1 HAVING sum(listens) >= {MIN_IN_FILE})
                       ORDER BY recording_mbid, week, release_mbid""",
            "plat": "SELECT week, user_id, count(*)::INTEGER AS listens FROM l GROUP BY ALL ORDER BY week, user_id",
        }
        for key, sql in queries.items():  # write to .tmp then rename, so a crash never leaves a partial file
            tmp = paths[key].with_suffix(".tmp")
            con.execute(f"COPY ({sql}) TO '{tmp.as_posix()}' (FORMAT parquet, COMPRESSION zstd)")
            tmp.replace(paths[key])
    finally:
        con.close()
    return file_no, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="scan only the first N files (for testing)")
    args = ap.parse_args()

    dump_id, url = latest_dump()
    index = parquet_index(dump_id, url)
    first = first_file_on_or_after(dump_id, url, index, PANEL_START)
    last = first_file_on_or_after(dump_id, url, index, PANEL_END_EXCL)
    rows = [index.iloc[i] for i in range(first, last + 1)][: args.limit]
    out_dir = INTERIM / f"weekly_scan_{dump_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Dump {dump_id}: scanning {len(rows)} files ({rows[0]['name']}..{rows[-1]['name']}) -> {out_dir}")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for n, (file_no, secs) in enumerate(ex.map(lambda r: scan_file(url, r, out_dir), rows), 1):
            if n % 10 == 0 or n == len(rows):
                mb = sum(p.stat().st_size for p in out_dir.glob("*.parquet")) / 1e6
                log(f"  {n}/{len(rows)} files, {time.time() - t0:.0f}s elapsed, {mb:,.0f} MB written")
    log("Done")


if __name__ == "__main__":
    main()
