"""Cached downloads of files given by URL (e.g. Zenodo records), verified against a checksum."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlparse

from seflag.errors import ConfigError, DataError

logger = logging.getLogger("seflag")
CHECKSUM = re.compile(r"^(md5|sha1|sha256):([0-9a-f]+)$")


def downloads_root() -> Path:
    return Path(os.environ.get("SEFLAG_HOME", Path.home() / ".cache" / "seflag")).expanduser() / "downloads"


def file_checksum(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def filename_from_url(url: str) -> str:
    """'https://zenodo.org/api/records/1/files/a.json/content' -> 'a.json'."""
    parts: list[str] = [p for p in unquote(urlparse(url).path).split("/") if p]
    if len(parts) >= 2 and parts[-1] == "content":  # Zenodo API links
        return parts[-2]
    return parts[-1] if parts else "download"


def download(url: str, checksum: str | None = None) -> Path:
    """Downloads ``url`` once into the cache and returns the local path.

    ``checksum`` (e.g. ``md5:0ce8...`` as listed on Zenodo, or ``sha256:...``) is verified after the download;
    a cached file is reused only if it still matches.
    """
    algorithm, expected = None, None
    if checksum is not None:
        match = CHECKSUM.match(checksum.lower())
        if not match:
            raise ConfigError(f"Invalid checksum {checksum!r}; expected e.g. 'md5:<hex>' or 'sha256:<hex>'.")
        algorithm, expected = match.groups()
    key: str = hashlib.sha256(url.encode()).hexdigest()[:16]
    target: Path = downloads_root() / key / filename_from_url(url)
    if target.is_file() and (expected is None or file_checksum(target, algorithm) == expected):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial: Path = target.with_name(target.name + ".part")
    logger.info("Downloading %s", url)
    request = urllib.request.Request(url, headers={"User-Agent": "seflag"})
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out, length=1 << 20)
    except OSError as e:
        partial.unlink(missing_ok=True)
        raise DataError(f"Download of {url} failed: {e}") from e
    if expected is not None:
        actual: str = file_checksum(partial, algorithm)
        if actual != expected:
            partial.unlink(missing_ok=True)
            raise DataError(f"Checksum mismatch for {url}: expected {algorithm}:{expected}, got {algorithm}:{actual}")
    partial.replace(target)
    return target
