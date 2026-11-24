"""Runs lamonpy for :mod:`seflag.backends.lamon` in lamonpy's own Python environment.

This script is executed by the interpreter of that environment, which has lamonpy but not SEFLAG installed. It must
therefore only use the standard library and lamonpy, and stay compatible with Python 3.8, the newest Python for
which lamonpy 0.2.0 publishes wheels.

Protocol: a JSON request on stdin, ``{"texts": [...], "dict_path": ..., "tagger_path": ..., "beam_size": ...,
"workers": ...}``; a JSON response on stdout, ``{"lamonpy": <version>, "python": <version>, "results": [...]}``
with one list of ``[start, end, lemma, tag]`` (character offsets into the text, Perseus-style tag) per text.
Both are UTF-8, independently of the locale (which matters on Windows).
"""

import json
import sys


def main():
    request = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    try:
        from lamonpy import Lamon
    except ImportError as e:
        sys.stderr.write(f"lamonpy is not installed in this environment ({sys.executable}): {e}\n")
        return 3
    try:
        from importlib.metadata import version
        lamonpy_version = version("lamonpy")
    except Exception:  # e.g. PackageNotFoundError for a lamonpy without package metadata
        lamonpy_version = None

    paths = {key: request[key] for key in ("dict_path", "tagger_path") if request.get(key)}
    lamon = Lamon(**paths)
    options = {"tag_style": "perseus"}
    if request.get("beam_size"):
        options["beam_size"] = request["beam_size"]
    texts = request["texts"]
    workers = request.get("workers") or 1
    if workers > 1 and hasattr(lamon, "tag_multi"):
        candidates = list(lamon.tag_multi(texts, num_workers=workers, **options))
    else:
        candidates = [lamon.tag(text, **options) for text in texts]
    # each text yields its beam of (score, tagged sequence) candidates, best first
    results = [[[start, end, lemma, tag] for start, end, lemma, tag in beam[0][1]] if beam else []
               for beam in candidates]
    response = {"lamonpy": lamonpy_version, "python": sys.version.split()[0], "results": results}
    sys.stdout.buffer.write(json.dumps(response, ensure_ascii=False).encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
