"""Build the cleaned release-impact dataset: one row per (artist, release group).

Inputs:  data/interim/releases_raw.parquet (fetch_releases.py), data/processed/*_week.parquet (build_panels.py),
         data/interim/weekly_scan_<dump>/ (per-user rows, for retention and audience features)
Outputs: data/processed/release_dataset.parquet, data/processed/release_dataset_quality.md

Time is measured in Friday-start weeks. Week 0 is the week containing the release date (T0). Features use
weeks -12..-1 only; targets use weeks 0..11. See docs/release_dataset_dictionary.md for every column.

    python scripts/build_release_dataset.py [--skip-enrich]
"""
import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from release_impact.config import INTERIM, PANEL_END_EXCL, PANEL_START, PRE_WEEKS, POST_WEEKS, PROCESSED  # noqa: E402
from release_impact.enrich import (artist_metadata, label_class, release_group_metadata,  # noqa: E402
                                   release_group_recordings)
from release_impact.lb_dump import log  # noqa: E402

STUDIO_TYPES = ["Album", "EP", "Single"]
MIN_PRE_LISTENERS = 20     # mean weekly unique listeners over weeks -12..-1
SMOOTH = 5                 # added to numerator and denominator of log ratios
NEARBY_WEEKS = (-2, 3)     # another release by the artist in this window -> not "clean"
BUMP_MIN_LIFT = 1.2        # decay is only defined for releases with at least this lift
MIN_NEW_LISTENERS = 30     # retention is flagged reliable at or above this many first-time listeners
LEAD_SINGLE_DAYS = 365     # tracks first released up to this many days before count as pre-released
# Development split from the model plan: train through 2023, validate on 2024, final test 2025 onward
SPLITS = [("train", pd.Timestamp("2024-01-01")), ("val", pd.Timestamp("2025-01-01")), ("test", pd.Timestamp("2100-01-01"))]
N_USER_BUCKETS = 8
LAST_WEEK = PANEL_END_EXCL - pd.Timedelta(days=7)
K = np.arange(-PRE_WEEKS, POST_WEEKS)  # week offsets in the series matrices


def friday_week(dates):
    """Friday that starts the release's week. Thursday dates are treated as Friday: they are almost
    always Friday-midnight drops dated in an earlier time zone (e.g. Red (Taylor's Version), 2021-11-11)."""
    dates = dates + pd.to_timedelta((dates.dt.dayofweek == 3).astype(int), unit="D")
    return (dates - pd.to_timedelta((dates.dt.dayofweek - 4) % 7, unit="D")).dt.normalize()


def log_ratio(post, pre):
    return np.log((post + SMOOTH) / (pre + SMOOTH))


def series_matrix(long, rids, col):
    """(len(rids), 24) array of a weekly column by week offset, 0 where the artist had no row."""
    pos = pd.Series(np.arange(len(rids)), index=rids)
    m = np.zeros((len(rids), len(K)))
    sub = long[long["k"].between(K[0], K[-1])]
    m[pos[sub["rid"]].to_numpy(), (sub["k"] - K[0]).to_numpy()] = sub[col].to_numpy()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-enrich", action="store_true", help="skip the LB/MB metadata calls")
    args = ap.parse_args()
    quality = {}
    con = duckdb.connect()
    tmp = INTERIM / "duckdb_tmp"
    tmp.mkdir(exist_ok=True)
    con.execute(f"SET temp_directory = '{tmp.as_posix()}'; SET memory_limit = '8GB'")
    P = {k: (PROCESSED / f"{k}.parquet").as_posix() for k in
         ("artist_week", "platform_week", "recording_week", "release_recordings")}
    scan_dir = sorted(INTERIM.glob("weekly_scan_*"))[-1].as_posix()

    # ---------------------------------------------------------------- 1. Release groups
    rel = pd.read_parquet(INTERIM / "releases_raw.parquet")
    quality["release groups fetched (2020-01 to 2026-10)"] = len(rel)
    rel = rel[rel["release_group_primary_type"].isin(STUDIO_TYPES) & rel["release_group_secondary_type"].isna()]
    rel = rel.sort_values("release_date").drop_duplicates("release_group_mbid")
    rel["artist_mbid"] = rel["artist_mbids"].map(lambda a: a[0] if len(a) else None)
    rel["featured_mbids"] = rel["artist_mbids"].map(lambda a: list(a[1:]))
    rel = rel.dropna(subset=["artist_mbid"]).reset_index(drop=True)
    rel["t0_week"] = friday_week(rel["release_date"])
    quality["studio album/EP/single release groups with an artist"] = len(rel)

    # Release history (all studio releases since 2020, before this one)
    rel = rel.sort_values(["artist_mbid", "release_date", "release_group_mbid"]).reset_index(drop=True)
    g = rel.groupby("artist_mbid")
    rel["n_prior_releases"] = g.cumcount()
    rel["days_since_last_release"] = (rel["release_date"] - g["release_date"].shift()).dt.days

    # Nearby releases by the same artist (contaminate the before/after windows)
    nb = rel[["artist_mbid", "t0_week", "release_group_mbid"]]
    pairs = nb.merge(nb, on="artist_mbid", suffixes=("", "_o"))
    pairs = pairs[pairs["release_group_mbid"] != pairs["release_group_mbid_o"]]
    dk = (pairs["t0_week_o"] - pairs["t0_week"]).dt.days // 7
    count_in = lambda lo, hi: rel["release_group_mbid"].map(  # noqa: E731
        pairs[dk.between(lo, hi)].groupby("release_group_mbid").size()).fillna(0).astype(int)
    rel["n_other_releases_nearby"] = count_in(*NEARBY_WEEKS)
    rel["prev_release_in_baseline"] = count_in(-PRE_WEEKS, -1) > 0   # baseline already contains another bump
    rel["next_release_in_outcome"] = count_in(0, 3) > 0              # outcome window contains another release

    cand = rel[(rel["t0_week"] >= PANEL_START + pd.Timedelta(weeks=PRE_WEEKS))
               & (rel["t0_week"] + pd.Timedelta(weeks=POST_WEEKS - 1) <= LAST_WEEK)].copy()
    cand = cand.reset_index(drop=True)
    cand["rid"] = np.arange(len(cand))
    quality[f"... released {cand['release_date'].min():%Y-%m-%d} to {cand['release_date'].max():%Y-%m-%d} "
            f"(12 weeks of panel before and after)"] = len(cand)
    log(f"{len(cand):,} candidate releases")

    # ---------------------------------------------------------------- 2. Weekly series, eligibility
    con.register("cand", cand[["rid", "artist_mbid", "t0_week"]].assign(t0_week=lambda d: d["t0_week"].dt.date))
    long = con.sql(f"""
        SELECT c.rid, ((a.week - c.t0_week) // 7)::INTEGER AS k, a.listens, a.listeners, a.repeat_listeners, a.new_listeners
        FROM cand c JOIN '{P['artist_week']}' a
          ON a.artist_mbid = c.artist_mbid
         AND a.week BETWEEN c.t0_week - INTERVAL {PRE_WEEKS * 7} DAY AND c.t0_week + INTERVAL {POST_WEEKS * 7 - 7} DAY
    """).df()
    L = series_matrix(long, cand["rid"], "listeners")
    pre = L[:, :PRE_WEEKS].mean(axis=1)
    keep = pre >= MIN_PRE_LISTENERS
    quality[f"... with >= {MIN_PRE_LISTENERS} mean weekly listeners in the 12 weeks before (eligible)"] = int(keep.sum())
    ds = cand[keep].reset_index(drop=True)
    rids = ds["rid"].to_numpy()
    long = long[long["rid"].isin(rids)]
    S = {c: series_matrix(long, rids, c) for c in ("listens", "listeners", "repeat_listeners", "new_listeners")}
    L = S["listeners"]
    pre_idx, w03, w811 = slice(0, PRE_WEEKS), slice(PRE_WEEKS, PRE_WEEKS + 4), slice(PRE_WEEKS + 8, PRE_WEEKS + 12)
    log(f"{len(ds):,} eligible releases")

    # ---------------------------------------------------------------- 3. Targets
    pre_mean, post4, late = L[:, pre_idx].mean(1), L[:, w03].mean(1), L[:, w811].mean(1)
    ds["pre_listeners_12w"] = pre_mean
    ds["post_listeners_4w"] = post4
    ds["lift_4w"] = post4 / pre_mean
    ds["log_lift_4w"] = log_ratio(post4, pre_mean)
    ds["abs_change_4w"] = post4 - pre_mean
    ds["sustained"] = (late / pre_mean >= 1.2)
    ds["late_lift_8_11w"] = late / pre_mean
    # Decay (survival target): only for releases with a bump. Counted from the peak in weeks 0-3 to the first
    # later week back within 20% of baseline; censored if that never happens by week 11.
    post = L[:, PRE_WEEKS:]
    peak = post[:, :4].argmax(1)
    back = (post <= 1.2 * pre_mean[:, None]) & (np.arange(POST_WEEKS)[None, :] > peak[:, None])
    has_bump = ds["lift_4w"].to_numpy() >= BUMP_MIN_LIFT
    ds["decay_weeks"] = np.where(has_bump, np.where(back.any(1), back.argmax(1), POST_WEEKS), np.nan)
    ds["decay_censored"] = pd.array(np.where(has_bump, ~back.any(1), None), dtype="boolean")

    # ---------------------------------------------------------------- 4. Listening features (weeks -12..-1)
    Lp = L[:, pre_idx]
    x = np.arange(PRE_WEEKS) - (PRE_WEEKS - 1) / 2
    ds["pre_listener_slope_12w"] = (Lp - Lp.mean(1, keepdims=True)) @ x / (x @ x) / np.maximum(pre_mean, 1)
    ds["pre_listener_cv_12w"] = Lp.std(1) / np.maximum(pre_mean, 1)
    ds["pre_listener_peak_12w"] = Lp.max(1)
    ds["pre_growth_4v8"] = Lp[:, -4:].mean(1) / np.maximum(Lp[:, :-4].mean(1), 1)
    ds["pre_listens_per_listener"] = S["listens"][:, pre_idx].sum(1) / np.maximum(Lp.sum(1), 1)
    ds["pre_repeat_listener_ratio"] = S["repeat_listeners"][:, pre_idx].sum(1) / np.maximum(Lp.sum(1), 1)
    ds["pre_new_listener_rate_4w"] = S["new_listeners"][:, PRE_WEEKS - 4:PRE_WEEKS].sum(1) / np.maximum(Lp[:, -4:].sum(1), 1)

    con.register("ds", ds[["rid", "artist_mbid", "t0_week", "release_mbid", "release_group_mbid", "release_date"]]
                 .assign(t0_week=lambda d: d["t0_week"].dt.date))
    plat = con.sql(f"""
        SELECT d.rid, avg(p.active_users) FILTER (p.week < d.t0_week) AS pre_active,
               arg_max(p.active_users, p.week) FILTER (p.week < d.t0_week) AS platform_active_users_prev_week
        FROM ds d JOIN '{P['platform_week']}' p ON p.week BETWEEN d.t0_week - INTERVAL {PRE_WEEKS * 7} DAY AND d.t0_week
        GROUP BY d.rid""").df().set_index("rid")
    ds["platform_active_users_prev_week"] = ds["rid"].map(plat["platform_active_users_prev_week"])
    ds["pre_listener_share"] = pre_mean / ds["rid"].map(plat["pre_active"]).to_numpy()

    # Artist history: lift of the previous eligible release whose 12 post weeks ended before T0
    ds = ds.sort_values(["artist_mbid", "t0_week"]).reset_index(drop=True)
    prev_lift, mean_prev = np.full(len(ds), np.nan), np.full(len(ds), np.nan)
    for _, idx in ds.groupby("artist_mbid").indices.items():
        t0s, lifts = ds["t0_week"].to_numpy()[idx], ds["log_lift_4w"].to_numpy()[idx]
        for j, i in enumerate(idx):
            done = t0s[:j] + np.timedelta64(POST_WEEKS * 7, "D") <= t0s[j]
            if done.any():
                prev_lift[i], mean_prev[i] = lifts[:j][done][-1], lifts[:j][done].mean()
    ds["prev_release_log_lift"], ds["mean_prev_log_lift"] = prev_lift, mean_prev
    ds["has_prior_release"] = ds["n_prior_releases"] > 0

    # ---------------------------------------------------------------- 5. Release content (tracks)
    # Two sources for "which recordings are on this release":
    #  - MusicBrainz recording search (all editions, with each recording's first release date): albums and EPs
    #  - the dump's listens on the release edition fresh-releases lists: singles, and the fallback.
    #    Listeners often play other editions, so this misses tracks on albums (hence MusicBrainz for those).
    art_rel = rel[rel["artist_mbid"].isin(ds["artist_mbid"])][["artist_mbid", "release_group_mbid", "release_mbid", "release_date"]]
    con.register("art_rel", art_rel)
    rr = con.sql(f"""SELECT a.release_group_mbid, r.recording_mbid
                     FROM art_rel a JOIN '{P['release_recordings']}' r USING (release_mbid)""").df()
    ds["n_tracks_heard"] = ds["release_group_mbid"].map(rr.groupby("release_group_mbid").size()).fillna(0).astype(int)
    long_form = ds["release_group_primary_type"].isin(["Album", "EP"])
    ds["tracks_source"] = "dump"
    ds["n_pre_released_tracks"], ds["days_since_lead_single"] = np.nan, np.nan
    if not args.skip_enrich:
        log("Release-group recordings (MusicBrainz search, albums and EPs) ...")
        mbr = release_group_recordings(ds.loc[long_form, "release_group_mbid"], log=log)
        mbr = mbr.merge(ds[["release_group_mbid", "release_date"]], on="release_group_mbid")
        first = pd.to_datetime(mbr["recording_first_release_date"].where(
            mbr["recording_first_release_date"].str.len() == 10), errors="coerce")
        lead_days = (mbr["release_date"] - first).dt.days
        # Pre-released: first out 2+ days before this release (1 day of slack for time zones), within a year
        pre = mbr[(lead_days >= 2) & (lead_days <= LEAD_SINGLE_DAYS)].assign(lead_days=lead_days)
        by_rg = pre.groupby("release_group_mbid")
        has_mb = ds["release_group_mbid"].isin(mbr["release_group_mbid"])
        ds.loc[has_mb, "tracks_source"] = "musicbrainz"
        ds.loc[has_mb, "n_pre_released_tracks"] = ds.loc[has_mb, "release_group_mbid"].map(
            by_rg["recording_mbid"].nunique()).fillna(0)
        ds.loc[has_mb, "days_since_lead_single"] = ds.loc[has_mb, "release_group_mbid"].map(by_rg["lead_days"].max())
        # Track listens use every recording MusicBrainz lists for the release group, plus the dump mapping
        rr = pd.concat([rr, mbr[["release_group_mbid", "recording_mbid"]]]).drop_duplicates()
    # Singles (and everything when MusicBrainz is skipped): earlier releases by the artist sharing a recording
    rr2 = rr.merge(art_rel[["release_group_mbid", "artist_mbid", "release_date"]], on="release_group_mbid")
    shared = rr2.merge(rr2, on=["artist_mbid", "recording_mbid"], suffixes=("", "_e"))
    gap = (shared["release_date"] - shared["release_date_e"]).dt.days
    shared = shared[(gap > 1) & (gap <= LEAD_SINGLE_DAYS)].assign(gap=gap)
    by_rg = shared.groupby("release_group_mbid")
    dump_rows = ds["tracks_source"] == "dump"
    ds.loc[dump_rows, "n_pre_released_tracks"] = ds.loc[dump_rows, "release_group_mbid"].map(
        by_rg["recording_mbid"].nunique()).fillna(0)
    ds.loc[dump_rows, "days_since_lead_single"] = ds.loc[dump_rows, "release_group_mbid"].map(by_rg["gap"].max())

    # Track-level listens: this release's recordings vs the rest of the catalog
    con.register("rr", rr)
    tl = con.sql(f"""
        SELECT d.rid, ((w.week - d.t0_week) // 7)::INTEGER AS k, sum(w.listens) AS rel_listens
        FROM ds d JOIN rr ON rr.release_group_mbid = d.release_group_mbid
        JOIN '{P['recording_week']}' w ON w.recording_mbid = rr.recording_mbid
         AND w.week BETWEEN d.t0_week - INTERVAL {PRE_WEEKS * 7} DAY AND d.t0_week + INTERVAL {POST_WEEKS * 7 - 7} DAY
        GROUP BY ALL""").df()
    R = series_matrix(tl, ds["rid"], "rel_listens")
    A = series_matrix(long, ds["rid"], "listens")
    ds["pre_single_listens_4w"] = R[:, PRE_WEEKS - 4:PRE_WEEKS].sum(1)
    cat = np.clip(A - R, 0, None)
    ds["catalog_log_lift_4w"] = log_ratio(cat[:, w03].mean(1), cat[:, pre_idx].mean(1))

    # ---------------------------------------------------------------- 6. Collaborators and competition
    feat = ds[["rid", "featured_mbids", "t0_week"]].explode("featured_mbids").dropna()
    ds["n_featured_artists"] = ds["featured_mbids"].map(len)
    ds["max_featured_pre_listeners"] = np.nan
    if len(feat):
        con.register("feat", feat.assign(t0_week=lambda d: d["t0_week"].dt.date))
        fl = con.sql(f"""
            SELECT f.rid, f.featured_mbids, sum(a.listeners) / {PRE_WEEKS} AS l
            FROM feat f JOIN '{P['artist_week']}' a ON a.artist_mbid = f.featured_mbids
             AND a.week BETWEEN f.t0_week - INTERVAL {PRE_WEEKS * 7} DAY AND f.t0_week - INTERVAL 7 DAY
            GROUP BY ALL""").df()
        ds["max_featured_pre_listeners"] = ds["rid"].map(fl.groupby("rid")["l"].max())
    ds.loc[ds["n_featured_artists"] > 0, "max_featured_pre_listeners"] = ds["max_featured_pre_listeners"].fillna(0)

    top = con.sql(f"""
        SELECT week + INTERVAL 7 DAY AS rel_week, artist_mbid FROM (
            SELECT week, artist_mbid, row_number() OVER (PARTITION BY week ORDER BY listeners DESC) AS rk
            FROM '{P['artist_week']}') WHERE rk <= 100""").df()
    top["rel_week"] = pd.to_datetime(top["rel_week"])
    rel_top = rel.merge(top, left_on=["t0_week", "artist_mbid"], right_on=["rel_week", "artist_mbid"])
    comp = rel_top.groupby("t0_week")["artist_mbid"].apply(set)
    ds["competing_top100_releases"] = [
        len(comp.get(t, set()) - {a}) for t, a in zip(ds["t0_week"], ds["artist_mbid"])]

    # ---------------------------------------------------------------- 7. Audience features and retention (per-user rows)
    log("Per-user features (retention, top-listener share, heavy users) ...")
    con.execute(f"""CREATE OR REPLACE TABLE heavy AS
        WITH uq AS (
            SELECT user_id, CAST(date_trunc('quarter', week) AS DATE) AS q, sum(listens) AS n
            FROM read_parquet('{scan_dir}/*_plat.parquet') GROUP BY 1, 2)
        SELECT user_id, q FROM (SELECT *, percent_rank() OVER (PARTITION BY q ORDER BY n) AS pr FROM uq)
        WHERE pr >= 0.9""")
    parts = []
    for b in range(N_USER_BUCKETS):
        con.execute(f"""CREATE OR REPLACE TABLE e AS
            SELECT rid, artist_mbid, t0_week FROM ds WHERE hash(artist_mbid) % {N_USER_BUCKETS} = {b}""")
        con.execute(f"""CREATE OR REPLACE TABLE u AS
            SELECT artist_mbid, user_id, week, sum(listens) AS n FROM read_parquet('{scan_dir}/*_ua.parquet')
            WHERE hash(artist_mbid) % {N_USER_BUCKETS} = {b} AND artist_mbid IN (SELECT artist_mbid FROM e)
            GROUP BY ALL""")
        con.execute("CREATE OR REPLACE TABLE fs AS SELECT artist_mbid, user_id, min(week) AS fw FROM u GROUP BY ALL")
        parts.append(con.sql(f"""
            WITH newu AS (
                SELECT e.rid, e.artist_mbid, e.t0_week, fs.user_id FROM e JOIN fs
                  ON fs.artist_mbid = e.artist_mbid AND fs.fw BETWEEN e.t0_week AND e.t0_week + INTERVAL 21 DAY),
            ret AS (
                SELECT DISTINCT n.rid, n.user_id FROM newu n JOIN u
                  ON u.artist_mbid = n.artist_mbid AND u.user_id = n.user_id
                 AND u.week BETWEEN n.t0_week + INTERVAL 56 DAY AND n.t0_week + INTERVAL 77 DAY),
            retention AS (
                SELECT n.rid, count(*) AS new_listeners_4w, count(r.user_id) AS retained_new_listeners
                FROM newu n LEFT JOIN ret r USING (rid, user_id) GROUP BY n.rid),
            pre AS (
                SELECT e.rid, e.t0_week, u.user_id, sum(u.n) AS n FROM e JOIN u
                  ON u.artist_mbid = e.artist_mbid AND u.week >= e.t0_week - INTERVAL {PRE_WEEKS * 7} DAY AND u.week < e.t0_week
                GROUP BY ALL),
            ranked AS (SELECT *, row_number() OVER (PARTITION BY rid ORDER BY n DESC) AS rk FROM pre),
            audience AS (
                SELECT r.rid,
                       sum(r.n) FILTER (r.rk <= 10) / sum(r.n) AS pre_top10_listener_share,
                       avg(CASE WHEN h.user_id IS NULL THEN 0 ELSE 1 END) AS heavy_user_share
                FROM ranked r LEFT JOIN heavy h
                  ON h.user_id = r.user_id AND h.q = CAST(date_trunc('quarter', r.t0_week) - INTERVAL 3 MONTH AS DATE)
                GROUP BY r.rid)
            SELECT e.rid, a.pre_top10_listener_share, a.heavy_user_share,
                   coalesce(t.new_listeners_4w, 0) AS new_listeners_4w, coalesce(t.retained_new_listeners, 0) AS retained_new_listeners
            FROM e LEFT JOIN audience a USING (rid) LEFT JOIN retention t USING (rid)""").df())
        log(f"  user bucket {b + 1}/{N_USER_BUCKETS}")
    users = pd.concat(parts).set_index("rid")
    for c in users.columns:
        ds[c] = ds["rid"].map(users[c])
    ds["new_listener_retention"] = ds["retained_new_listeners"] / ds["new_listeners_4w"].where(ds["new_listeners_4w"] > 0)
    ds["retention_reliable"] = ds["new_listeners_4w"] >= MIN_NEW_LISTENERS

    # ---------------------------------------------------------------- 8. Metadata (current snapshots)
    if not args.skip_enrich:
        log("Artist metadata (LB) ...")
        am = artist_metadata(ds["artist_mbid"], log=log)
        ds = ds.merge(am, on="artist_mbid", how="left")
        log("Release metadata (MusicBrainz search, 1 req/s) ...")
        rm = release_group_metadata(ds["release_group_mbid"], log=log)
        ds = ds.merge(rm, on="release_group_mbid", how="left")
        # labels is NaN for release groups the MusicBrainz search didn't return
        ds["label_class"] = [label_class([n for n in (lab if isinstance(lab, str) else "").split("; ") if n], name)
                             for lab, name in zip(ds["labels"], ds["artist_credit_name"])]

    # ---------------------------------------------------------------- 9. Calendar, flags, split
    ds["release_type"] = ds["release_group_primary_type"]
    ds["release_dow"] = ds["release_date"].dt.dayofweek
    ds["is_friday_release"] = ds["release_dow"] == 4
    ds["release_month"] = ds["release_date"].dt.month
    ds["is_q4"] = ds["release_month"] >= 10
    ds["is_clean"] = ds["n_other_releases_nearby"] == 0
    ds["split"] = pd.cut(ds["release_date"], [pd.Timestamp("1900-01-01")] + [s[1] for s in SPLITS],
                         labels=[s[0] for s in SPLITS], right=False).astype(str)
    train_artists = set(ds.loc[ds["split"] == "train", "artist_mbid"])
    ds["artist_in_train"] = ds["artist_mbid"].isin(train_artists)
    # 1/sqrt(releases by the artist in the same split), so prolific artists don't decide the fit
    ds["sample_weight"] = 1 / np.sqrt(ds.groupby(["split", "artist_mbid"])["artist_mbid"].transform("size"))

    ds = ds.drop(columns=["rid", "artist_mbids", "release_group_primary_type", "release_group_secondary_type"])
    ds["featured_mbids"] = ds["featured_mbids"].map("; ".join)
    first = ["artist_mbid", "artist_credit_name", "release_group_mbid", "release_mbid", "release_name", "release_type",
             "release_date", "t0_week", "split", "is_clean", "artist_in_train"]
    ds = ds[first + [c for c in ds.columns if c not in first]].sort_values(["release_date", "artist_mbid"])
    out = PROCESSED / "release_dataset.parquet"
    ds.to_parquet(out, index=False)
    log(f"Saved {out}: {len(ds):,} rows x {ds.shape[1]} columns")
    write_quality(ds, quality)


def md(df):
    """DataFrame -> Markdown table (avoids a tabulate dependency)."""
    df = df.reset_index()
    rows = [[f"{v:,.3f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v) for v in r] for r in df.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * df.shape[1]]
                     + ["| " + " | ".join(r) + " |" for r in rows])


def write_quality(ds, quality):
    lines = ["# Release dataset quality report", "",
             f"Built {pd.Timestamp.now():%Y-%m-%d %H:%M}. One row per (artist, release group).", "",
             "## Row counts", "", "| Step | Rows |", "|---|---|"]
    lines += [f"| {k} | {v:,} |" for k, v in quality.items()]
    lines += [f"| ... of which clean (no other release by the artist in weeks {NEARBY_WEEKS[0]}..+{NEARBY_WEEKS[1]}) | {int(ds['is_clean'].sum()):,} |",
              "", "## Split", "",
              ds.groupby("split").agg(rows=("is_clean", "size"), clean=("is_clean", "sum"),
                                      artists=("artist_mbid", "nunique"), artist_in_train=("artist_in_train", "mean"))
              .round(3).pipe(md), "",
              "## By release type", "",
              ds.groupby("release_type").agg(rows=("is_clean", "size"), median_lift=("lift_4w", "median"),
                                             sustained=("sustained", "mean"), median_retention=("new_listener_retention", "median"))
              .round(3).pipe(md), "",
              "## Targets", "",
              ds[["lift_4w", "log_lift_4w", "abs_change_4w", "late_lift_8_11w", "decay_weeks", "new_listener_retention",
                  "catalog_log_lift_4w"]].describe().round(3).pipe(md), "",
              f"- `sustained` positive rate: {ds['sustained'].mean():.1%}",
              f"- releases with a bump (lift >= {BUMP_MIN_LIFT}, so `decay_weeks` is defined): {(ds['lift_4w'] >= BUMP_MIN_LIFT).mean():.1%}; "
              f"of those censored: {ds['decay_censored'].mean():.1%}",
              f"- `retention_reliable` (>= {MIN_NEW_LISTENERS} first-time listeners): {ds['retention_reliable'].mean():.1%}",
              f"- `prev_release_in_baseline`: {ds['prev_release_in_baseline'].mean():.1%}; "
              f"`next_release_in_outcome`: {ds['next_release_in_outcome'].mean():.1%}",
              f"- track source: {ds['tracks_source'].value_counts().to_dict()}", "",
              "## Coverage (share of rows with a value)", "", "| Column | Non-null |", "|---|---|"]
    lines += [f"| `{c}` | {v:.1%} |" for c, v in ds.notna().mean().items() if v < 1]
    (PROCESSED / "release_dataset_quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
