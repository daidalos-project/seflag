"""Decoding of token-level entity tags (BIO, IOB1, BIOES, BILOU, IO) into entity spans."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from seflag.data.records import EntitySpan

MARKERS: frozenset[str] = frozenset({"B", "I", "E", "S", "L", "U"})
STARTS: frozenset[str] = frozenset({"B", "S", "U"})
ENDS: frozenset[str] = frozenset({"E", "S", "L", "U"})


class TagScheme(BaseModel):
    """How entity tags are written, e.g. ``B-PER`` (prefix) or ``PRS-B`` (suffix)."""

    model_config = ConfigDict(extra="forbid")
    position: Literal["prefix", "suffix"] = Field(
        "prefix", description="Where the B/I/E/S marker is: 'prefix' (B-PER) or 'suffix' (PER-B).")
    separator: str = Field("-", description="Separator between marker and entity type.")
    outside: list[str] = Field(default_factory=lambda: ["O", "0", "_", ""],
                               description="Tags meaning 'not part of an entity'.")

    def parse(self, tag: str) -> tuple[str, str] | None:
        """'B-PER' -> ('B', 'PER'); 'PER' (IO scheme) -> ('I', 'PER'); outside tags -> None."""
        if tag in self.outside:
            return None
        if self.position == "prefix":
            marker, sep, label = tag.partition(self.separator)
        else:
            label, sep, marker = tag.rpartition(self.separator)
        if sep and marker in MARKERS and label:
            return marker, label
        return "I", tag


def decode(tags: list[str], scheme: TagScheme) -> list[EntitySpan]:
    """Converts tags to spans with conlleval semantics: an I/E/L tag without a matching open entity starts one."""
    spans: list[EntitySpan] = []
    current: EntitySpan | None = None
    for i, tag in enumerate(tags, start=1):
        parsed = scheme.parse(tag)
        if parsed is None:
            current = None
            continue
        marker, label = parsed
        if current is None or marker in STARTS or label != current.label:
            current = EntitySpan(first=i, last=i, label=label)
            spans.append(current)
        else:
            current.last = i
        if marker in ENDS:
            current = None
    return spans
