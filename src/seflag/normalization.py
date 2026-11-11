"""Named string normalizers that can be referenced from configuration files.

Normalization is a frequent source of hidden differences between published evaluation results
(e.g. u/v spelling in Latin, Unicode composition in Greek). In SEFLAG, every normalization step
is explicit and recorded in the results.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable

Normalizer = Callable[[str], str]


def _strip_diacritics(text: str) -> str:
    decomposed: str = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", "".join(c for c in decomposed if not unicodedata.combining(c)))


NORMALIZERS: dict[str, tuple[Normalizer, str]] = {
    "nfc": (lambda s: unicodedata.normalize("NFC", s), "Unicode canonical composition (NFC)"),
    "nfd": (lambda s: unicodedata.normalize("NFD", s), "Unicode canonical decomposition (NFD)"),
    "nfkc": (lambda s: unicodedata.normalize("NFKC", s), "Unicode compatibility composition (NFKC)"),
    "lowercase": (str.lower, "Lowercase all characters"),
    "v_to_u": (lambda s: s.replace("v", "u").replace("V", "U"), "Latin: replace v/V with u/U"),
    "j_to_i": (lambda s: s.replace("j", "i").replace("J", "I"), "Latin: replace j/J with i/I"),
    "strip_diacritics": (_strip_diacritics, "Remove combining marks (accents, breathings, macrons)"),
    "strip_homonym_digits": (
        lambda s: re.sub(r"(?<=\D)#?\d+$", "", s),
        "Remove trailing homonym numbers from lemmas, e.g. 'sum1' or 'sum#1' (Morpheus) -> 'sum'",
    ),
}


def get_pipeline(names: list[str]) -> Normalizer:
    """Composes the named normalizers (applied in the given order) into a single function."""
    unknown: list[str] = [n for n in names if n not in NORMALIZERS]
    if unknown:
        raise ValueError(f"Unknown normalizer(s) {unknown}. Available: {', '.join(NORMALIZERS)}")
    functions: list[Normalizer] = [NORMALIZERS[n][0] for n in names]

    def pipeline(text: str) -> str:
        for fn in functions:
            text = fn(text)
        return text

    return pipeline
