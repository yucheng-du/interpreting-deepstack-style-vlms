"""Bounded parallel ranges with a resumable, contiguous on-disk prefix.

Callers must serialize writers for each destination (the model downloader already
holds a Hub file lock). This module uses only the standard library.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request

_MAX_CHUNK_BYTES = 8 * 1024**2
_MAX_ATTEMPTS = 3
_TIMEOUT_SECONDS = 60
_RETRYABLE_HTTP_CODES = {408, 429, 500, 502, 503, 504}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _range_bytes(url, start, end, expected_size, allow_modelscope_range_200=False):
    size = end - start + 1
    parsed_url = urllib.parse.urlsplit(url)
    allow_200 = (allow_modelscope_range_200 is True
                 and parsed_url.scheme == "https"
                 and parsed_url.hostname == "modelscope.cn")
    separator = "&" if "?" in url else "?"
    # Distinct resolver URLs avoid proxies reusing a different Range response.
    request_url = f"{url}{separator}download=true&range_start={start}&range_end={end}"
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        body = None  # Release any short response before allocating its retry.
        try:
            request = urllib.request.Request(
                request_url,
                headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
            )
            with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
                if response.status != 206 and not (response.status == 200 and allow_200):
                    raise ValueError(f"Unexpected HTTP status {response.status} for byte range")
                value = response.headers.get("Content-Range", "")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value)
                if (not match
                        or tuple(map(int, match.groups())) != (start, end, expected_size)):
                    raise ValueError(f"Invalid Content-Range for {start}-{end}: {value!r}")
                length = response.headers.get("Content-Length")
                if response.status == 200 and length is None:
                    raise ValueError("ModelScope HTTP 200 ranges require an exact Content-Length")
                if length is not None and (not length.isdigit() or int(length) != size):
                    raise ValueError(f"Invalid Content-Length for {start}-{end}: {length!r}")
                if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise ValueError("Encoded byte ranges are not supported")
                body = response.read(size + 1)
                if len(body) > size:
                    raise ValueError(f"Oversized range body for {start}-{end}")
                if len(body) < size:
                    raise OSError(f"Truncated range body for {start}-{end}")
                return body
        except (OSError, http.client.IncompleteRead) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code not in _RETRYABLE_HTTP_CODES:
                raise
            if attempt == _MAX_ATTEMPTS:
                raise OSError(f"Range {start}-{end} failed after {_MAX_ATTEMPTS} attempts") from error
            print(f"  Retrying range {start}-{end}: {type(error).__name__} "
                  f"(attempt {attempt + 1}/{_MAX_ATTEMPTS})", flush=True)
            time.sleep(attempt)


def download_weight(url, destination, expected_sha256, expected_size,
                    range_workers=2, chunk_bytes=8 * 1024**2,
                    allow_modelscope_range_200=False):
    """Return the verified SHA256, resuming ``destination.with_suffix('.partial')``.

    At most two adjacent ranges are fetched per batch. Nothing from a failed
    batch is appended. A successful batch is appended in offset order and synced
    before fetching the next; interruption during append still leaves a prefix,
    never a sparse file. Existing prefix bytes are checked by the final SHA256.

    Buffers are limited to two 8 MiB responses per call (about 32 MiB across the
    model downloader's two outer file workers), and discarded between batches.
    HTTP 200 is accepted only with explicit opt-in for an HTTPS modelscope.cn
    request URL, still requiring exact Content-Range and Content-Length headers.
    """
    if type(expected_size) is not int or expected_size <= 0:
        raise ValueError("expected_size must be a positive integer")
    if type(range_workers) is not int or not 1 <= range_workers <= 2:
        raise ValueError("range_workers must be 1 or 2")
    if type(chunk_bytes) is not int or not 1 <= chunk_bytes <= _MAX_CHUNK_BYTES:
        raise ValueError("chunk_bytes must be between 1 byte and 8 MiB")
    if type(allow_modelscope_range_200) is not bool:
        raise ValueError("allow_modelscope_range_200 must be an explicit boolean")
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("expected_sha256 must be a 64-character SHA256")
    expected_sha256 = expected_sha256.lower()
    destination = Path(destination)
    partial = destination.with_suffix(".partial")
    if partial == destination:
        raise ValueError("The final destination cannot itself have the .partial suffix")
    if destination.exists():
        if destination.stat().st_size != expected_size or _sha256(destination) != expected_sha256:
            raise ValueError(f"Existing destination size/SHA256 mismatch: {destination}")
        return expected_sha256
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise ValueError(f"Existing partial exceeds expected size: {partial}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=range_workers) as pool:
        while offset < expected_size:
            ranges = [(start, min(start + chunk_bytes, expected_size) - 1)
                      for start in range(offset, min(offset + range_workers * chunk_bytes, expected_size), chunk_bytes)]
            # map preserves range order, even when the later response arrives first.
            chunks = list(pool.map(
                lambda pair: _range_bytes(url, *pair, expected_size, allow_modelscope_range_200), ranges))
            with partial.open("ab") as handle:
                if os.fstat(handle.fileno()).st_size != offset:
                    raise RuntimeError("Partial changed during a batch; callers must serialize writers")
                for index in range(len(chunks)):
                    handle.write(chunks[index])
                handle.flush()
                os.fsync(handle.fileno())
            offset = ranges[-1][1] + 1
            del chunks  # Do not retain a previous batch while buffering the next.
            print(f"  Downloaded {offset / 1e6:.1f}/{expected_size / 1e6:.1f} MB", flush=True)
    if partial.stat().st_size != expected_size:
        raise ValueError(f"Downloaded size mismatch: {partial}")
    actual_sha256 = _sha256(partial)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Downloaded SHA256 mismatch: {partial}")
    partial.replace(destination)
    return actual_sha256
