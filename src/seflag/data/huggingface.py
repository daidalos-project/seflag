from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field

from seflag.data.base import DatasetAdapter
from seflag.data.conllu import read_conllu_files, split_from_filename
from seflag.data.files import collect_files, file_hashes
from seflag.data.records import TOKEN_FIELDS, SentenceRecord, TokenRecord
from seflag.data.tags import TagScheme
from seflag.data.tags import decode as decode_tags
from seflag.errors import ConfigError, DataError, MissingDependencyError

LOCAL_BUILDERS: set[str] = {"parquet", "json", "csv", "arrow"}
SENTENCE_FIELDS: tuple[str, ...] = ("sent_id", "text", "subset")
SEFLAG_SENTENCE_FIELDS: tuple[str, ...] = ("comments", "multiword_tokens", "empty_nodes", "entities")
# default column names for datasets with one list column per annotation layer (as e.g. 'universal_dependencies')
COLUMN_DEFAULTS: dict[str, str] = {
    "form": "tokens", "lemma": "lemmas", "upos": "upos", "xpos": "xpos", "feats": "feats",
    "head": "heads", "deprel": "deprels", "deps": "deps", "misc": "misc", "ner": "ner_tags", "sent_id": "sent_id", "text": "text", "subset": "subset",
}


def _require_datasets():
    try:
        import datasets
    except ImportError as e:
        raise MissingDependencyError(
            "Hugging Face datasets support is not installed. Run: pip install 'seflag[hf]'") from e
    return datasets


def _require_hub():
    try:
        import huggingface_hub
    except ImportError as e:
        raise MissingDependencyError(
            "Hugging Face Hub support is not installed. Run: pip install 'seflag[hf]'") from e
    return huggingface_hub


def _clean(value: Any) -> Any:
    return None if value == "_" else value


class HuggingFaceDataset(DatasetAdapter):
    """A dataset on the Hugging Face Hub, in one of two formats.

    * ``conllu``: the repository contains CoNLL-U files, which are downloaded and parsed directly.
    * ``table``: anything ``datasets.load_dataset`` can read, one row per sentence. Either seflag's own layout
      (``tokens`` = list of token dicts, see ``seflag convert-conllu``) or parallel list columns
      (``tokens``, ``lemmas``, ``upos``, ...; rename them via ``columns``).

    ``format: auto`` picks ``conllu`` if the repository contains .conllu files.
    """

    description = "Hugging Face Hub dataset: CoNLL-U files in a repository, or a table loadable with `datasets`."

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")
        repo: str = Field(description="Hub repository id (e.g. 'daidalos-project/...') or a builder like 'parquet'.")
        revision: str | None = Field(None, description="Git revision; pin a commit hash for reproducible results.")
        split: str = Field("test", description="Split to evaluate on (for CoNLL-U: inferred from UD file names).")
        format: Literal["auto", "conllu", "table"] = Field("auto", description="See class documentation.")
        files: list[str] | None = Field(
            None, description="conllu: files in the repository to use (default: all .conllu files of the split).")
        subset_by_file: bool = Field(True, description="conllu: report results per file in addition to the total.")
        name: str | None = Field(None, description="table: dataset configuration (subset) name.")
        data_files: str | list[str] | dict[str, Any] | None = Field(None, description="table: for load_dataset().")
        columns: dict[str, str] = Field(
            default_factory=dict,
            description="table: mapping from record fields (form, lemma, upos, ..., ner, sent_id, text, subset) "
                        "to dataset column names.")
        tag_scheme: TagScheme = Field(
            default_factory=TagScheme, description="table: encoding of a token-level 'ner' tag column.")

    def __init__(self, config: Any, evaluation):
        super().__init__(config, evaluation)
        unknown: set[str] = set(config.columns) - set(COLUMN_DEFAULTS)
        if unknown:
            raise ConfigError(f"Unknown record field(s) in 'columns': {sorted(unknown)}; "
                              f"valid fields: {', '.join(COLUMN_DEFAULTS)}")
        self._format: str | None = None
        self._dataset = None
        self._snapshot: Path | None = None

    # --- format detection -------------------------------------------------------------------------------

    @property
    def format(self) -> str:
        if self._format is None:
            if self.config.format != "auto":
                self._format = self.config.format
            elif self.config.repo in LOCAL_BUILDERS:
                self._format = "table"
            else:
                files: list[str] = _require_hub().list_repo_files(
                    self.config.repo, repo_type="dataset", revision=self.config.revision)
                self._format = "conllu" if any(f.endswith(".conllu") for f in files) else "table"
        return self._format

    def records(self) -> Iterator[SentenceRecord]:
        return self._conllu_records() if self.format == "conllu" else self._table_records()

    # --- CoNLL-U files in a Hub repository --------------------------------------------------------------

    def snapshot(self) -> Path:
        """Downloads (or reuses from the Hugging Face cache) the repository's CoNLL-U files."""
        if self._snapshot is None:
            patterns: list[str] = self.config.files or ["*.conllu", "**/*.conllu"]
            self._snapshot = Path(_require_hub().snapshot_download(
                self.config.repo, repo_type="dataset", revision=self.config.revision, allow_patterns=patterns))
        return self._snapshot

    def conllu_files(self) -> list[Path]:
        root: Path = self.snapshot()
        if self.config.files:
            return collect_files([root / f for f in self.config.files], "*.conllu")
        files: list[Path] = [p for p in sorted(root.rglob("*.conllu")) if split_from_filename(p) == self.config.split]
        if not files:
            raise DataError(f"No CoNLL-U files for split '{self.config.split}' in {self.config.repo}")
        return files

    def _conllu_records(self) -> Iterator[SentenceRecord]:
        return read_conllu_files(self.conllu_files(), self.config.subset_by_file)

    # --- tabular datasets -----------------------------------------------------------------------------

    def dataset(self):
        if self._dataset is None:
            datasets = _require_datasets()
            data_files = self.config.data_files
            if isinstance(data_files, (str, list)) and self.config.repo in LOCAL_BUILDERS:
                data_files = [str(self.evaluation.resolve_path(f))
                              for f in ([data_files] if isinstance(data_files, str) else data_files)]
            self._dataset = datasets.load_dataset(
                self.config.repo, name=self.config.name, split=self.config.split,
                revision=self.config.revision, data_files=data_files)
        return self._dataset

    def _table_records(self) -> Iterator[SentenceRecord]:
        ds = self.dataset()
        columns: dict[str, str] = {**COLUMN_DEFAULTS, **self.config.columns}
        token_column: str = columns["form"]
        if token_column not in ds.column_names:
            raise ConfigError(f"Dataset has no token column '{token_column}'. Columns: {ds.column_names}")
        sentence_columns: dict[str, str] = {f: columns[f] for f in SENTENCE_FIELDS if columns[f] in ds.column_names}
        # parallel list columns, possibly with ClassLabel-encoded values (e.g. UPOS ids)
        layer_columns: dict[str, str] = {f: columns[f] for f in TOKEN_FIELDS
                                         if f != "form" and columns[f] in ds.column_names}
        decoders: dict[str, Any] = {}
        for field, col in layer_columns.items():
            inner = getattr(ds.features[col], "feature", None)
            if inner is not None and hasattr(inner, "int2str"):
                decoders[field] = inner.int2str
        ner_column: str | None = columns["ner"] if columns["ner"] in ds.column_names else None
        ner_decoder = None
        if ner_column:
            inner = getattr(ds.features[ner_column], "feature", None)
            ner_decoder = inner.int2str if inner is not None and hasattr(inner, "int2str") else None
        for row in ds:
            tokens_value: list = row[token_column]
            if tokens_value and isinstance(tokens_value[0], dict):  # seflag layout: list of token dicts
                yield SentenceRecord.from_dict({
                    **{k: v for k, v in row.items() if k in SEFLAG_SENTENCE_FIELDS},
                    "tokens": [{k: v for k, v in t.items() if k in TOKEN_FIELDS} for t in tokens_value],
                    **{f: row[c] for f, c in sentence_columns.items()},
                })
                continue
            n: int = len(tokens_value)
            values: dict[str, list] = {}
            for field, col in layer_columns.items():
                column_values: list = row[col] or [None] * n
                if len(column_values) != n:
                    raise DataError(f"Row {row.get(sentence_columns.get('sent_id', ''))!r}: column '{col}' "
                                    f"has {len(column_values)} entries for {n} tokens.")
                decode = decoders.get(field)
                values[field] = [_clean(decode(v) if decode and v is not None else v) for v in column_values]
            tokens = [TokenRecord(form=form, **{f: values[f][i] for f in values}) for i, form in enumerate(tokens_value)]
            entities = None
            if ner_column:
                tags: list = row[ner_column] or []
                if ner_decoder is not None:
                    tags = [ner_decoder(t) for t in tags]
                entities = decode_tags([str(t) for t in tags], self.config.tag_scheme)
            yield SentenceRecord(tokens=tokens, entities=entities, **{f: row[c] for f, c in sentence_columns.items()})

    # --- provenance -----------------------------------------------------------------------------------

    def fingerprint(self) -> dict[str, Any]:
        info: dict[str, Any] = {"repo": self.config.repo, "revision": self.config.revision,
                                "split": self.config.split, "format": self.format}
        if self.format == "conllu":
            # snapshot directories are named after the commit hash
            info["resolved_revision"] = self.snapshot().name
            info["files"] = file_hashes(self.conllu_files(), self.snapshot())
            return info
        info.update(name=self.config.name, fingerprint=self.dataset()._fingerprint)
        if self.config.repo not in LOCAL_BUILDERS:
            try:
                info["resolved_revision"] = _require_hub().HfApi().dataset_info(
                    self.config.repo, revision=self.config.revision).sha
            except Exception:  # offline; the datasets fingerprint is still recorded
                pass
        return info
