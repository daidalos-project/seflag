"""Typed configuration: enums instead of strings, typed component Configs, validation when loading."""

from pathlib import Path

import pytest

from seflag import (
    BackendName,
    DatasetSource,
    Language,
    Layer,
    Normalizer,
    TaskType,
    Upos,
    load_config,
    load_dataset,
    load_matrix,
    register,
    registry,
)
from seflag.backends.flair import FlairBackend
from seflag.backends.morpheus import MorpheusBackend
from seflag.config import EvaluationOptions, ModelSpec
from seflag.data.conllu import ConlluDataset
from seflag.data.huggingface import HuggingFaceDataset
from seflag.enums import FallbackLemma, PredictionLayer, Tagset
from seflag.errors import ConfigError
from seflag.matrix import is_matrix
from seflag.tasks.lemmatization import LemmatizationTask
from seflag.tasks.pos_tagging import PosTaggingTask
from seflag.tests.conftest import LowercaseBackend

EXAMPLES = Path(__file__).parent.parent / "examples"


def test_members_behave_like_their_values():
    assert Language.LATIN == "la" and str(Language.ANCIENT_GREEK) == "grc" and f"{TaskType.NER}" == "ner"
    assert {Normalizer.NFC: 1}["nfc"] == 1 and "lemma" in {Layer.LEMMA}  # same hash as the value


def test_typed_python_config_equals_dictionary_config(conllu_dir, fake_backend):
    typed = load_config({
        "language": Language.LATIN,
        "task": LemmatizationTask.Config(exclude_upos=[Upos.PUNCT]),
        "dataset": ConlluDataset.Config(path=str(conllu_dir)),
        "model": ModelSpec(backend=fake_backend),
        "options": EvaluationOptions(label_normalization=[Normalizer.V_TO_U, Normalizer.LOWERCASE]),
        "output": {"dir": None},
    })
    plain = load_config({
        "language": "la", "task": {"type": "lemmatization", "exclude_upos": ["PUNCT"]},
        "dataset": {"source": "conllu", "path": str(conllu_dir)}, "model": {"backend": fake_backend},
        "options": {"label_normalization": ["v_to_u", "lowercase"]}, "output": {"dir": None},
    })
    assert typed.model_dump(mode="json") == plain.model_dump(mode="json")
    assert typed.task.type is TaskType.LEMMATIZATION and typed.dataset.source is DatasetSource.CONLLU
    assert typed.options.label_normalization == [Normalizer.V_TO_U, Normalizer.LOWERCASE]


@pytest.mark.parametrize("change, message", [
    ({"language": "latin"}, "Input should be 'la' or 'grc'"),
    ({"task": "lemmatisation"}, "did you mean 'lemmatization'"),
    ({"dataset": {"source": "conll"}}, "did you mean 'conllu'"),
    ({"model": {"backend": "spacey"}}, "did you mean 'spacy'"),
    ({"options": {"input_normalization": ["nfcc"]}}, "Input should be 'nfc'"),
    ({"task": {"type": "lemmatization", "exclude_upos": ["PUNKT"]}}, "Input should be 'ADJ'"),
    ({"task": {"type": "pos_tagging", "tagest": "upos"}}, "'tagest': did you mean 'tagset'"),
    ({"model": {"backend": "morpheus", "unknown": "forms"}}, "Input should be 'empty' or 'form'"),
])
def test_typos_are_rejected_when_loading(conllu_dir, change, message):
    data = {"language": "la", "task": "lemmatization", "dataset": {"source": "conllu", "path": str(conllu_dir)},
            "model": {"backend": "morpheus"}, **change}
    with pytest.raises(ConfigError, match=message):
        load_config(data)


def test_auto_still_means_automatic():
    assert MorpheusBackend.Config(language="auto").language is None
    assert FlairBackend.Config(name="x", layer="auto").layer is None
    assert HuggingFaceDataset.Config(repo="x", format="auto").format is None
    assert FlairBackend.Config(name="x", layer=PredictionLayer.UPOS).layer is PredictionLayer.UPOS
    assert MorpheusBackend.Config(unknown="form").unknown is FallbackLemma.FORM


def test_plugin_names_and_typed_configs_of_plugins(conllu_dir, monkeypatch):
    monkeypatch.setattr(registry, "_REGISTERED", {kind: {} for kind in registry.BUILTINS})

    class Plugin(LowercaseBackend):
        pass

    with pytest.raises(ConfigError, match="unknown backend 'plugin'"):
        load_config({"language": "la", "task": "lemmatization", "dataset": {"source": "conllu", "path": "x"},
                     "model": {"backend": "plugin"}})
    register("backends", "plugin", Plugin)
    config = load_config({"language": "la", "task": "lemmatization", "dataset": {"source": "conllu", "path": "x"},
                          "model": Plugin.Config()})
    assert config.model.backend == "plugin" and not isinstance(config.model.backend, BackendName)
    with pytest.raises(ConfigError, match="not the Config of a registered task"):
        load_config({"language": "la", "task": EvaluationOptions(), "dataset": {"source": "conllu", "path": "x"},
                     "model": {"backend": "plugin"}})


def test_dataset_section_typed_and_layers(conllu_dir):
    dataset = load_dataset(ConlluDataset.Config(path=str(conllu_dir)), language=Language.LATIN)
    assert dataset.docs(layers=[Layer.LEMMA, "upos"], max_sentences=1)[0][0].pos_ == "NOUN"
    with pytest.raises(ValueError, match="'lemmas' is not a valid Layer"):
        dataset.docs(layers=["lemmas"])
    with pytest.raises(ConfigError, match="language"):
        load_dataset(ConlluDataset.Config(path=str(conllu_dir)), language="latin")


def test_matrix_entries_accept_typed_configs(conllu_dir, fake_backend):
    matrix = load_matrix({
        "task": TaskType.POS_TAGGING, "language": Language.LATIN, "output": {"dir": None},
        "dataset": ConlluDataset.Config(path=str(conllu_dir)),
        "matrix": {"models": [
            {"label": "upos", "model": ModelSpec(backend=fake_backend),
             "task": PosTaggingTask.Config(tagset=Tagset.UPOS)},
            {"label": "greek", "language": Language.ANCIENT_GREEK, "model": {"backend": fake_backend}},
        ]},
    })
    assert [c.config.task.params for c in matrix.cells] == [{"tagset": "upos"}, {}]
    assert matrix.source["matrix"]["models"][0]["task"] == {"tagset": "upos"}  # stored as plain values


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.yaml")), ids=lambda p: p.name)
def test_example_configurations_are_valid(path):
    (load_matrix if is_matrix(path) else load_config)(path)
