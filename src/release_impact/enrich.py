"""Batch metadata for artists (LB metadata API) and release groups (MusicBrainz search), cached on disk.

Both are *current* snapshots, not as-of-release values. Genres, artist type and labels rarely change
after release, so we use them and note it in the data dictionary.
"""
import hashlib
import json
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import RAW, USER_AGENT

CACHE = RAW / "api_cache"

PARENT_GENRES = {  # MB genre -> parent genre; first keyword match wins, so specific groups come first
    "hip-hop": ["hip hop", "rap", "trap", "drill", "grime"],
    "r&b/soul": ["r&b", "soul", "funk"],
    "metal": ["metal", "deathcore"],
    "punk/emo": ["punk", "hardcore", "emo"],
    "electronic": ["electronic", "house", "techno", "edm", "dubstep", "drum and bass", "trance", "ambient",
                   "synthwave", "hyperpop", "idm", "electro", "breakbeat"],
    "latin": ["latin", "reggaeton", "cumbia", "salsa", "bachata", "corrido", "regional mexican"],
    "k-pop/j-pop": ["k-pop", "j-pop", "j-rock", "city pop", "anime"],
    "reggae/afro": ["reggae", "dancehall", "afrobeat", "amapiano"],
    "country": ["country", "americana", "bluegrass"],
    "folk": ["folk", "singer-songwriter", "acoustic"],
    "jazz/blues": ["jazz", "blues", "swing"],
    "classical/soundtrack": ["classical", "orchestral", "soundtrack", "opera", "baroque"],
    "pop": ["pop", "dance"],
    "rock": ["rock", "indie", "alternative", "grunge", "shoegaze"],
}
MAJOR_LABEL_KEYWORDS = ["universal", "sony", "warner", "columbia", "rca", "epic", "atlantic", "interscope",
                        "republic", "capitol", "def jam", "island", "geffen", "motown", "polydor", "parlophone",
                        "elektra", "virgin", "emi", "arista", "300 entertainment", "aftermath"]

lb = requests.Session()
lb.headers.update({"User-Agent": USER_AGENT})
lb.mount("https://", HTTPAdapter(max_retries=Retry(
    total=8, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=None)))
mb = requests.Session()
mb.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
_mb_last = 0.0


def _get(session, url, **kwargs):
    """session.get that also survives dropped connections and timeouts (waits up to a minute, 8 tries)."""
    for attempt in range(8):
        try:
            return session.get(url, **kwargs)
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(min(60, 5 * 2 ** attempt))
    return session.get(url, **kwargs)


def _cached(key, fetch):
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def _batch_key(prefix, ids):
    return f"{prefix}/{hashlib.sha1(','.join(sorted(ids)).encode()).hexdigest()}"


def parent_genre(genres):
    for genre in genres:
        for parent, keywords in PARENT_GENRES.items():
            if any(k in genre for k in keywords):
                return parent
    return None


def label_class(label_names, artist_name):
    names = [n.lower() for n in label_names]
    if any(k in n for n in names for k in MAJOR_LABEL_KEYWORDS):
        return "major"
    if names and all(n in ("[no label]", (artist_name or "").lower()) for n in names):
        return "self-released"
    return "independent" if names else "unknown"


def artist_metadata(mbids, batch=50, log=print):
    """One row per artist: name, type, area, begin_year, genres (MB genres by vote), primary_genre."""
    mbids = sorted(set(mbids))
    rows = []
    for start in range(0, len(mbids), batch):
        ids = mbids[start:start + batch]

        def fetch():
            for attempt in range(5):
                resp = _get(lb, "https://api.listenbrainz.org/1/metadata/artist/", timeout=60,
                              params={"artist_mbids": ",".join(ids), "inc": "tag"})
                if resp.status_code == 429:
                    time.sleep(int(resp.headers.get("X-RateLimit-Reset-In", 5)) + 1)
                    continue
                resp.raise_for_status()
                return resp.json()
            resp.raise_for_status()

        for a in _cached(_batch_key("lb_artist_meta", ids), fetch):
            tags = sorted((a.get("tag") or {}).get("artist") or [], key=lambda t: -t.get("count", 0))
            genres = [t["tag"] for t in tags if t.get("genre_mbid")]
            rows.append({
                "artist_mbid": a["artist_mbid"], "artist_name": a.get("name"), "artist_type": a.get("type"),
                "area": a.get("area"), "begin_year": a.get("begin_year"),
                "mb_genres": "; ".join(genres[:3]), "n_genres": len(genres), "primary_genre": parent_genre(genres),
            })
        if (start // batch) % 20 == 0:
            log(f"    artist metadata {min(start + batch, len(mbids)):,}/{len(mbids):,}")
    return pd.DataFrame(rows).drop_duplicates("artist_mbid")


def _mb_search_releases(rgids):
    global _mb_last
    query = " OR ".join(f"rgid:{r}" for r in rgids)
    releases, offset = [], 0
    while True:
        for attempt in range(6):
            time.sleep(max(0.0, 1.1 - (time.time() - _mb_last)))
            _mb_last = time.time()
            resp = _get(mb, "https://musicbrainz.org/ws/2/release", timeout=60,
                          params={"query": query, "fmt": "json", "limit": 100, "offset": offset})
            if resp.status_code not in (429, 500, 502, 503, 504):
                break
            time.sleep(3 * (attempt + 1))
        resp.raise_for_status()
        data = resp.json()
        releases += data.get("releases", [])
        offset += 100
        if offset >= data.get("count", 0) or offset >= 500:  # cap: a few huge reissue families
            return releases


def _mb_search(entity, query, max_results=500):
    """Page through a MusicBrainz search (100 per page), 1 request/second, retrying on 503."""
    global _mb_last
    items, offset = [], 0
    while True:
        for attempt in range(6):
            time.sleep(max(0.0, 1.1 - (time.time() - _mb_last)))
            _mb_last = time.time()
            resp = _get(mb, f"https://musicbrainz.org/ws/2/{entity}", timeout=60,
                          params={"query": query, "fmt": "json", "limit": 100, "offset": offset})
            if resp.status_code not in (429, 500, 502, 503, 504):
                break
            time.sleep(3 * (attempt + 1))
        resp.raise_for_status()
        data = resp.json()
        items += data.get(f"{entity}s", [])
        offset += 100
        if offset >= data.get("count", 0) or offset >= max_results:
            return items


def release_group_recordings(rgids, batch=10, log=print):
    """Recordings on each release group (any edition), with each recording's first release date.

    One row per (release_group_mbid, recording_mbid). A recording whose first release date is before the
    release group's date was pre-released (a lead single); one dated after it was added by a later edition.
    """
    rgids = sorted(set(rgids))
    rows = []
    for start in range(0, len(rgids), batch):
        ids = rgids[start:start + batch]
        recs = _cached(_batch_key("mb_rg_recordings", ids),
                       lambda: _mb_search("recording", " OR ".join(f"rgid:{r}" for r in ids), max_results=1000))
        wanted = set(ids)
        for rec in recs:
            for rel in rec.get("releases") or []:
                rg = (rel.get("release-group") or {}).get("id")
                if rg in wanted:
                    rows.append({"release_group_mbid": rg, "recording_mbid": rec["id"],
                                 "recording_first_release_date": rec.get("first-release-date")})
        if (start // batch) % 100 == 0:
            log(f"    release-group recordings {min(start + batch, len(rgids)):,}/{len(rgids):,}")
    df = pd.DataFrame(rows, columns=["release_group_mbid", "recording_mbid", "recording_first_release_date"])
    return df.drop_duplicates(["release_group_mbid", "recording_mbid"])


def release_group_metadata(rgids, batch=20, log=print):
    """One row per release group: n_tracks (median over its releases) and label names."""
    rgids = sorted(set(rgids))
    rows = []
    for start in range(0, len(rgids), batch):
        ids = rgids[start:start + batch]
        releases = _cached(_batch_key("mb_rg_search", ids), lambda: _mb_search_releases(ids))
        per_rg = {}
        for r in releases:
            rg = (r.get("release-group") or {}).get("id")
            if rg not in ids:
                continue
            d = per_rg.setdefault(rg, {"tracks": [], "labels": set()})
            if r.get("track-count"):
                d["tracks"].append(r["track-count"])
            for li in r.get("label-info") or []:
                if (li.get("label") or {}).get("name"):
                    d["labels"].add(li["label"]["name"])
        for rg, d in per_rg.items():
            rows.append({"release_group_mbid": rg,
                         "n_tracks": float(pd.Series(d["tracks"]).median()) if d["tracks"] else None,
                         "labels": "; ".join(sorted(d["labels"]))})
        if (start // batch) % 50 == 0:
            log(f"    release metadata {min(start + batch, len(rgids)):,}/{len(rgids):,}")
    return pd.DataFrame(rows, columns=["release_group_mbid", "n_tracks", "labels"])
