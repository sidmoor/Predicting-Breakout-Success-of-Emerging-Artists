"""Stream Parquet files out of the ListenBrainz Spark full dump with HTTP range requests.

The Spark dump is an uncompressed .tar of ~180 MB Parquet files in time order. We index the tar once
(name, data offset, size per member; cached CSV, ~25 min per dump) and then read individual columns
of individual files without downloading the ~250 GB archive.
"""
import csv
import io
import json
import re
import tarfile
import time

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import RAW, USER_AGENT

DUMP_ROOT = "https://data.metabrainz.org/pub/musicbrainz/listenbrainz/fullexport/"
CACHE_DIR = RAW / "lb_dumps"

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})
session.mount("https://", HTTPAdapter(max_retries=Retry(
    total=10, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=None)))


def latest_dump():
    """(dump_id, spark tar URL) of the newest full dump on the server."""
    listing = session.get(DUMP_ROOT, timeout=30).text
    dirs = sorted(set(re.findall(r'href="(listenbrainz-dump-(\d+)-\d{8}-\d+-full)/"', listing)), key=lambda d: int(d[1]))
    dump_dir, dump_id = dirs[-1]
    files = re.findall(r'href="([^"]+\.tar(?:\.zst)?)"', session.get(DUMP_ROOT + dump_dir + "/", timeout=30).text)
    return dump_id, f"{DUMP_ROOT}{dump_dir}/" + next(f for f in files if "spark-dump" in f)


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only view of bytes [start, start + size) of a remote file."""

    def __init__(self, url, start, size):
        self.url, self.start, self.size, self.pos = url, start, size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        first = self.start + self.pos
        resp = session.get(self.url, headers={"Range": f"bytes={first}-{first + n - 1}"}, timeout=120)
        resp.raise_for_status()
        buf[:n] = resp.content
        self.pos += n
        return n


def _next_header(url, offset):
    """(offset, TarInfo) of the next real member header, skipping pax headers."""
    buf = session.get(url, headers={"Range": f"bytes={offset}-{offset + 2047}"}, timeout=60).content
    pos = 0
    while pos + 512 <= len(buf):
        block = buf[pos:pos + 512]
        if block == b"\0" * 512:
            return None, None
        info = tarfile.TarInfo.frombuf(block, "utf-8", "surrogateescape")
        if info.type != tarfile.XHDTYPE:
            return offset + pos, info
        pos += 512 + (info.size + 511) // 512 * 512
    return _next_header(url, offset + pos)


def parquet_index(dump_id, url):
    """Parquet members of the dump (name, data_offset, size, file_no), building/resuming the cached index."""
    path = CACHE_DIR / f"spark_index_{dump_id}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("name,data_offset,size\n")
    if not path.read_text().rstrip().endswith("#complete"):
        done = pd.read_csv(path)
        offset = 0
        if len(done):
            last = done.iloc[-1]
            offset = int(last["data_offset"]) + (int(last["size"]) + 511) // 512 * 512
        with path.open("a", newline="") as f:
            writer = csv.writer(f)
            while True:
                offset, info = _next_header(url, offset)
                if info is None:
                    f.write("#complete\n")
                    break
                writer.writerow((info.name.split("/")[-1], offset + 512, info.size))
                f.flush()
                offset += 512 + (info.size + 511) // 512 * 512
    idx = pd.read_csv(path, comment="#")
    idx = idx[idx["name"].str.endswith(".parquet")].copy()
    idx["file_no"] = idx["name"].str.removesuffix(".parquet").astype(int)
    return idx.sort_values("file_no").reset_index(drop=True)


def open_member(url, row):
    return HTTPRangeFile(url, int(row["data_offset"]), int(row["size"]))


def file_time_ranges(dump_id, url, index, file_ids):
    """Min/max listened_at for the given files (cached JSON; reading listened_at is ~4% of a file)."""
    path = CACHE_DIR / f"spark_time_ranges_{dump_id}.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    for i in file_ids:
        key = str(int(index.at[i, "file_no"]))
        if key not in cache:
            col = pq.ParquetFile(open_member(url, index.iloc[i])).read(columns=["listened_at"])["listened_at"]
            mm = pc.min_max(col)
            cache[key] = [str(mm["min"].as_py()), str(mm["max"].as_py())]
            path.write_text(json.dumps(cache))
    return {i: tuple(pd.Timestamp(t) for t in cache[str(int(index.at[i, "file_no"]))]) for i in file_ids}


def first_file_on_or_after(dump_id, url, index, ts):
    """Index position of the first file with listens on/after ts (files are in time order)."""
    lo, hi = 0, len(index) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if file_time_ranges(dump_id, url, index, [mid])[mid][1] < ts:
            lo = mid + 1
        else:
            hi = mid
    return lo


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)
