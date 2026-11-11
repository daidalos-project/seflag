"""Tab-separated column files with one token per line, as used by many NER datasets (CoNLL-2003 style)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from pydantic import Field

from seflag.data.base import DatasetAdapter
from seflag.data.files import FileResolver, FileSource
from seflag.data.records import SentenceRecord, TokenRecord
from seflag.data.tags import TagScheme, decode
from seflag.errors import ConfigError, DataError

TOKEN_COLUMNS: tuple[str, ...] = ("form", "lemma", "upos", "xpos")


class ColumnsDataset(DatasetAdapter):
    """Token-per-line files, local or in a Hub repository; sentences are separated by empty lines.

    Example (Herodotos NER data, header ``Label<TAB>Word``, tags like ``PRS-B``)::

        dataset:
          source: columns
          path: data/herodotos
          columns: {form: Word, ner: Label}
          tag_scheme: {position: suffix}
    """

    description = "Token-per-line column files (CoNLL-2003-style NER data), local or in a Hub repository."

    class Config(FileSource):
        header: bool = Field(True, description="The first line of each file names the columns.")
        columns: dict[str, str | int] = Field(
            description="Record field -> column name (with header) or 0-based index: form (required), ner, "
                        "lemma, upos, xpos.")
        delimiter: str = Field("\t", description="Column separator.")
        tag_scheme: TagScheme = Field(default_factory=TagScheme, description="How the 'ner' column is encoded.")
        split_after: list[str] = Field(
            default_factory=list,
            description="Additionally end a sentence after these tokens, e.g. ['.'] for files without empty lines.")
        subset_by_file: bool = Field(True, description="Report results per file in addition to the total.")

    def __init__(self, config: Any, evaluation):
        super().__init__(config, evaluation)
        unknown: set[str] = set(config.columns) - {*TOKEN_COLUMNS, "ner"}
        if unknown or "form" not in config.columns:
            raise ConfigError(f"'columns' needs 'form' and may contain {', '.join((*TOKEN_COLUMNS, 'ner'))}; "
                              f"unknown: {sorted(unknown)}")
        self.resolver = FileResolver(config, evaluation)

    def files(self) -> list[Path]:
        return self.resolver.files()

    def _indices(self, header: list[str] | None, path: Path) -> dict[str, int]:
        indices: dict[str, int] = {}
        for field, column in self.config.columns.items():
            if isinstance(column, int):
                indices[field] = column
            elif header is None:
                raise ConfigError(f"Column '{column}' given by name, but 'header' is false.")
            elif column not in header:
                raise DataError(f"{path}: no column '{column}' in header {header}")
            else:
                indices[field] = header.index(column)
        return indices

    def _sentence(self, rows: list[tuple[int, list[str]]], indices: dict[str, int], path: Path, number: int,
                  subset: str | None) -> SentenceRecord:
        tokens: list[TokenRecord] = []
        tags: list[str] = []
        for line_no, cols in rows:
            if max(indices.values()) >= len(cols):
                raise DataError(f"{path}:{line_no}: expected at least {max(indices.values()) + 1} columns")
            values: dict[str, str | None] = {f: cols[i] for f, i in indices.items() if f != "ner"}
            for f in ("lemma", "upos", "xpos"):
                if values.get(f) in ("_", ""):
                    values[f] = None
            tokens.append(TokenRecord(**values))
            if "ner" in indices:
                tags.append(cols[indices["ner"]])
        entities = decode(tags, self.config.tag_scheme) if "ner" in indices else None
        return SentenceRecord(tokens=tokens, sent_id=f"{path.stem}-{number}", subset=subset, entities=entities)

    def records(self) -> Iterator[SentenceRecord]:
        for path in self.files():
            subset: str | None = path.stem if self.config.subset_by_file else None
            with path.open(encoding="utf-8") as f:
                lines: list[str] = [line.rstrip("\r\n") for line in f]
            header: list[str] | None = None
            if self.config.header and lines:
                header, lines = lines[0].split(self.config.delimiter), lines[1:]
            indices: dict[str, int] = self._indices(header, path)
            rows: list[tuple[int, list[str]]] = []
            number: int = 0
            first_line: int = 2 if self.config.header else 1
            for line_no, line in enumerate(lines, start=first_line):
                if line.strip():
                    cols: list[str] = line.split(self.config.delimiter)
                    rows.append((line_no, cols))
                    if cols[indices["form"]] not in self.config.split_after:
                        continue
                if rows:
                    number += 1
                    yield self._sentence(rows, indices, path, number, subset)
                    rows = []
            if rows:
                yield self._sentence(rows, indices, path, number + 1, subset)

    def fingerprint(self) -> dict[str, Any]:
        return self.resolver.fingerprint()
