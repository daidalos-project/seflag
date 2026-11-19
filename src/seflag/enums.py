"""The fixed vocabularies of SEFLAG's configuration, as enums.

In Python, use the members instead of strings, so that typos are caught by your editor and by the validation::

    from seflag import load_config
    from seflag.enums import Language, Normalizer, TaskType

    config = load_config({"language": Language.LATIN, "task": TaskType.LEMMATIZATION, ...,
                          "options": {"label_normalization": [Normalizer.V_TO_U]}})

In YAML files, write the values (``language: la``); they are validated against the same enums. Every member is
also a ``str`` and compares equal to its value (``Language.LATIN == "la"``), and ``str()`` gives the value.

Names of tasks, dataset sources and backends are extensible (plugins add their own), so ``TaskType``,
``DatasetSource`` and ``BackendName`` list the built-in ones; any other name must be registered (see
:func:`seflag.register`). Vocabularies that are defined by the data, such as split names, dependency relations or
morphological features, remain strings.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class StrEnum(str, Enum):
    """An enum whose members are strings and print as their values (like ``enum.StrEnum`` in Python 3.11+)."""

    # hash like the value, so that members and their values are interchangeable as dictionary keys
    __hash__ = str.__hash__

    def __str__(self) -> str:
        return str(self.value)

    def __format__(self, spec: str) -> str:
        return format(str(self.value), spec)


def auto_to_none(value: Any) -> Any:
    """Before-validator for optional enum fields: the old spelling ``"auto"`` means "not set" (determined
    automatically)."""
    return None if value == "auto" else value


# --- evaluation -----------------------------------------------------------------------------------------------

class ComponentKind(StrEnum):
    """Kinds of components, e.g. for :func:`seflag.register`."""

    TASKS = "tasks"
    DATASETS = "datasets"
    BACKENDS = "backends"


class Language(StrEnum):
    """ISO 639-3 codes of the languages SEFLAG evaluates."""

    LATIN = "la"
    ANCIENT_GREEK = "grc"


class TaskType(StrEnum):
    """Built-in tasks (``task.type``)."""

    LEMMATIZATION = "lemmatization"
    POS_TAGGING = "pos_tagging"
    MORPHOLOGY = "morphology"
    DEPENDENCY_PARSING = "dependency_parsing"
    NER = "ner"
    WORD_SIMILARITY = "word_similarity"
    WORD_RETRIEVAL = "word_retrieval"
    SYNONYM_SELECTION = "synonym_selection"


class DatasetSource(StrEnum):
    """Built-in dataset sources (``dataset.source``)."""

    CONLLU = "conllu"
    HUGGINGFACE = "huggingface"
    COLUMNS = "columns"
    WORD_PAIRS = "word_pairs"
    WORD_CHOICES = "word_choices"


class BackendName(StrEnum):
    """Built-in model backends (``model.backend``)."""

    SPACY = "spacy"
    TRANSFORMERS = "transformers"
    FLAIR = "flair"
    MORPHEUS = "morpheus"
    GENSIM = "gensim"


class Normalizer(StrEnum):
    """String normalizers for ``options.input_normalization`` and ``options.label_normalization``."""

    NFC = "nfc"
    NFD = "nfd"
    NFKC = "nfkc"
    LOWERCASE = "lowercase"
    V_TO_U = "v_to_u"
    J_TO_I = "j_to_i"
    STRIP_DIACRITICS = "strip_diacritics"
    ELISION_TO_U02BC = "elision_to_u02bc"
    ELISION_TO_U2019 = "elision_to_u2019"
    ELISION_TO_APOSTROPHE = "elision_to_apostrophe"
    FINAL_SIGMA = "final_sigma"
    LUNATE_SIGMA = "lunate_sigma"
    STRIP_HOMONYM_DIGITS = "strip_homonym_digits"


class Tokenization(StrEnum):
    GOLD = "gold"


class SentenceSegmentation(StrEnum):
    GOLD = "gold"
    MODEL = "model"


class MultiwordTokens(StrEnum):
    SYNTACTIC_WORDS = "syntactic_words"


class Layer(StrEnum):
    """Annotation layers of the gold data that tasks require and Docs carry."""

    LEMMA = "lemma"
    UPOS = "upos"
    XPOS = "xpos"
    MORPH = "morph"
    DEPENDENCIES = "dependencies"
    ENTITIES = "entities"


class Upos(StrEnum):
    """Universal POS tags (Universal Dependencies)."""

    ADJ = "ADJ"
    ADP = "ADP"
    ADV = "ADV"
    AUX = "AUX"
    CCONJ = "CCONJ"
    DET = "DET"
    INTJ = "INTJ"
    NOUN = "NOUN"
    NUM = "NUM"
    PART = "PART"
    PRON = "PRON"
    PROPN = "PROPN"
    PUNCT = "PUNCT"
    SCONJ = "SCONJ"
    SYM = "SYM"
    VERB = "VERB"
    X = "X"


# --- task options -----------------------------------------------------------------------------------------------

class Tagset(StrEnum):
    """Which tag layer POS tagging compares."""

    UPOS = "upos"
    XPOS = "xpos"


class FeatureSelection(StrEnum):
    """Which morphological features are scored (or give an explicit list of feature names)."""

    UNIVERSAL = "universal"
    ALL = "all"
    MODEL = "model"


class ParsingMetric(StrEnum):
    UAS = "uas"
    LAS = "las"
    CLAS = "clas"
    MLAS = "mlas"
    BLEX = "blex"


class DeprelSubtypes(StrEnum):
    """Whether relation subtypes count (``obl:arg``) or are reduced to the universal relation (``obl``)."""

    IGNORE = "ignore"
    KEEP = "keep"


class OovPolicy(StrEnum):
    """Words outside an embedding model's vocabulary: skip them, or build vectors from fastText subwords."""

    SKIP = "skip"
    SUBWORDS = "subwords"


class RetrievalDirections(StrEnum):
    FORWARD = "forward"
    BOTH = "both"


class MissingCandidates(StrEnum):
    """Synonym selection: what to do with an item whose distractors are unknown to the model."""

    SKIP_ITEM = "skip_item"
    IGNORE = "ignore"


# --- dataset options -------------------------------------------------------------------------------------------

class RepoType(StrEnum):
    """Type of a Hugging Face Hub repository."""

    DATASET = "dataset"
    MODEL = "model"


class HubFormat(StrEnum):
    """How a Hub dataset is stored: CoNLL-U files or a table loadable with ``datasets``."""

    CONLLU = "conllu"
    TABLE = "table"


class LexicalFormat(StrEnum):
    JSON = "json"
    TSV = "tsv"
    CSV = "csv"


class PairField(StrEnum):
    """Fields of a word-pair item."""

    WORD1 = "word1"
    WORD2 = "word2"
    SCORE = "score"


class TagPosition(StrEnum):
    """Where an entity tag's marker stands: ``B-PER`` (prefix) or ``PER-B`` (suffix)."""

    PREFIX = "prefix"
    SUFFIX = "suffix"


# --- model options ----------------------------------------------------------------------------------------------

class PredictionLayer(StrEnum):
    """What a token classification model predicts."""

    ENTITIES = "entities"
    UPOS = "upos"
    XPOS = "xpos"


class EmbeddingFormat(StrEnum):
    FASTTEXT = "fasttext"
    WORD2VEC = "word2vec"
    WORD2VEC_TEXT = "word2vec_text"
    GENSIM = "gensim"


class MorpheusInstall(StrEnum):
    """Without a Morpheus installation: build one automatically, or raise an error."""

    AUTO = "auto"
    NEVER = "never"


class LemmaChoice(StrEnum):
    """Which lemma to use when Morpheus returns several."""

    FIRST = "first"
    LAST = "last"


class FallbackLemma(StrEnum):
    """Lemma for words that are not analysed: empty (scored as wrong) or the word form itself."""

    EMPTY = "empty"
    FORM = "form"
