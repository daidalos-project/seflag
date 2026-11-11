import pytest

from seflag.config import load_config
from seflag.errors import ConfigError

BASE = {"language": "la", "task": "lemmatization",
        "dataset": {"source": "conllu", "path": "data"}, "model": {"backend": "spacy", "name": "m"}}


def test_task_shorthand_and_params():
    config = load_config(BASE)
    assert config.task.type == "lemmatization"
    assert config.dataset.params == {"path": "data"}
    assert config.display_name == "lemmatization-la-m"


def test_relative_paths_resolve_against_config_file(tmp_path):
    (tmp_path / "eval.yaml").write_text(
        "language: la\ntask: lemmatization\ndataset: {source: conllu, path: data}\nmodel: {backend: spacy, name: m}\n")
    config = load_config(tmp_path / "eval.yaml")
    assert config.resolve_path("data") == tmp_path / "data"


def test_overrides_merge_sections():
    config = load_config({**BASE, "options": {"label_normalization": ["nfc"]}}, options={"max_sentences": 3})
    assert config.options.max_sentences == 3
    assert config.options.label_normalization == ["nfc"]


@pytest.mark.parametrize("bad", [
    {"options": {"label_normalization": ["nope"]}},
    {"options": {"tokenization": "raw"}},
    {"unknown_key": 1},
    {"language": None},
])
def test_invalid_configs(bad):
    with pytest.raises(ConfigError):
        load_config({**BASE, **bad})
