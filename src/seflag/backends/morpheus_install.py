"""Automatic installation of Morpheus: a pinned source release is downloaded and compiled in SEFLAG's cache.

Morpheus Perseids publishes no binaries, but its releases contain the precompiled stem library, so only the
analyser itself has to be compiled: this takes a few seconds and needs a C compiler (``cc``, or ``$CC``),
``make`` and ``ar`` (Linux: e.g. ``apt install build-essential``; macOS: ``xcode-select --install``). flex,
which the upstream build instructions list, is only needed for the tools that rebuild the stem library and is
not required. Windows is not supported; use WSL, or a Docker launcher (see :mod:`seflag.launcher`).

The installation goes to ``$SEFLAG_HOME/tools/morpheus-<version>`` (default ``~/.cache/seflag``) and is
reused afterwards; ``seflag cache purge`` deletes it (it is rebuilt on next use).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from seflag.data.download import download, seflag_home
from seflag.errors import MissingDependencyError, SeflagError

logger = logging.getLogger("seflag")

VERSION: str = "1.0.3"
URL: str = f"https://github.com/perseids-tools/morpheus-perseids/archive/refs/tags/v{VERSION}.tar.gz"
CHECKSUM: str = "sha256:4656cc0585046e47b909a6bbbe71225247b3ee3e1e1defb440f990a395a2c95b"
# directories in src/ and their make targets, in dependency order: the libraries the analyser links, then it
TARGETS: list[tuple[str, str]] = [("greeklib", "greeklib.a"), ("morphlib", "morphlib.a"), ("gkends", "gkends.a"),
                                  ("gkdict", "gkdict.a"), ("gener", "gener.a"), ("anal", "morpheus")]
# Morpheus is K&R-era C: gnu89 keeps implicit declarations legal, -fcommon allows its duplicate globals
# (GCC >= 10), and the -Wno-error flags keep newer compilers (GCC >= 14, clang >= 16) from rejecting it
CFLAGS: str = ("-std=gnu89 -fcommon -w -Wno-error=implicit-function-declaration -Wno-error=implicit-int "
               "-Wno-error=int-conversion -Wno-error=incompatible-pointer-types -Wno-error=return-type")
MARKER: str = "seflag-install.json"


def install_dir() -> Path:
    return seflag_home() / "tools" / f"morpheus-{VERSION}"


def installed() -> Path | None:
    """The automatic installation, if it exists and is complete."""
    root: Path = install_dir()
    return root if (root / MARKER).is_file() and (root / "bin" / "morpheus").is_file() else None


def ensure_installed() -> Path:
    """Returns the automatic Morpheus installation, downloading and compiling it first if necessary."""
    root: Path | None = installed()
    if root is not None:
        return root
    if sys.platform == "win32":
        raise MissingDependencyError(
            "Morpheus cannot be installed automatically on Windows. Run SEFLAG in WSL, or set 'path' to a "
            "Morpheus installation, e.g. inside Docker via a launcher (see the documentation of seflag.launcher).")
    compiler: str = os.environ.get("CC", "cc")
    missing: list[str] = [tool for tool in (compiler, "make", "ar") if shutil.which(tool) is None]
    if missing:
        hint: str = ("xcode-select --install" if sys.platform == "darwin"
                     else "e.g. sudo apt install build-essential (Debian/Ubuntu) or sudo dnf install gcc make")
        raise MissingDependencyError(
            f"Morpheus is not installed, and building it needs {', '.join(missing)} ({hint}). Alternatively, set "
            "'path' or $MORPHEUS_PATH to an existing installation.")
    target: Path = install_dir()
    logger.info("Morpheus not found: building Morpheus %s from source in %s (one-time, takes a few seconds)",
                VERSION, target)
    archive: Path = download(URL, CHECKSUM)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=f".{target.name}-") as tmp:
        build: Path = _extract(archive, Path(tmp))
        _compile(build / "src", compiler, target.parent / f"{target.name}-build.log")
        (build / "bin").mkdir(exist_ok=True)
        shutil.copy2(build / "src" / "anal" / "morpheus", build / "bin" / "morpheus")
        _smoke_test(build)
        shutil.rmtree(build / "src")  # only bin/ and stemlib/ are needed at runtime
        (build / MARKER).write_text(json.dumps({"version": VERSION, "url": URL, "checksum": CHECKSUM,
                                                "compiler": compiler, "cflags": CFLAGS}, indent=2) + "\n")
        try:
            build.rename(target)
        except OSError:
            if installed() is None:  # not a concurrent installation that finished first
                shutil.rmtree(target, ignore_errors=True)
                build.rename(target)
    logger.info("Morpheus %s installed in %s", VERSION, target)
    return target


def _extract(archive: Path, destination: Path) -> Path:
    with tarfile.open(archive) as tar:
        if hasattr(tarfile, "data_filter"):  # Python >= 3.12 and security backports
            tar.extractall(destination, filter="data")
        else:
            for member in tar.getmembers():
                resolved = (destination / member.name).resolve()
                if not resolved.is_relative_to(destination.resolve()) or member.issym() or member.islnk():
                    raise SeflagError(f"Unsafe path in Morpheus archive: {member.name}")
            tar.extractall(destination)
    tops: list[Path] = [p for p in destination.iterdir() if p.is_dir()]
    if len(tops) != 1 or not (tops[0] / "src").is_dir() or not (tops[0] / "stemlib").is_dir():
        raise SeflagError(f"Unexpected layout of the Morpheus archive {archive}")
    return tops[0]


def _compile(src: Path, compiler: str, log_path: Path) -> None:
    jobs: str = str(os.cpu_count() or 1)
    log: list[str] = []
    # the makefiles append their include path to CFLAGS (CFLAGS+=), so it is passed via the environment;
    # variables on the make command line override the hard-coded CC=gcc and LEX=flex of the makefiles
    env: dict[str, str] = {**os.environ, "CFLAGS": CFLAGS}
    for directory, make_target in TARGETS:
        args: list[str] = ["make", "-C", str(src / directory), f"-j{jobs}", make_target, f"CC={compiler}",
                           "LEX=false", "LEXLIB="]
        done = subprocess.run(args, capture_output=True, text=True, env=env, check=False)
        log += [" ".join(args), done.stdout, done.stderr]
        if done.returncode != 0:
            log_path.write_text("\n".join(log), encoding="utf-8")
            raise SeflagError(f"Building Morpheus failed in src/{directory} (exit code {done.returncode}). "
                              f"Build log: {log_path}\n{done.stderr[-1500:]}")


def _smoke_test(root: Path) -> None:
    """Analyses one Greek and one Latin word, which also catches stem library files the build cannot read."""
    for args, word, lemma in (([], "lo/gos", "lo/gos"), (["-L"], "amo", "amo")):
        done = subprocess.run([str(root / "bin" / "morpheus"), *args], input=word + "\n", capture_output=True,
                              text=True, cwd=root, env={**os.environ, "MORPHLIB": "stemlib"}, timeout=60, check=False)
        if f">{lemma}</hdwd>" not in done.stdout:
            raise SeflagError(f"The freshly built Morpheus does not analyse '{word}' correctly "
                              f"(exit code {done.returncode}): {(done.stdout + done.stderr)[:500]}")
