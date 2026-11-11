import pytest
from spacy.tokens import Doc
from spacy.vocab import Vocab

from seflag import load_config, registry, run
from seflag.backends.base import Backend
from seflag.errors import ConfigError

PREDICTED = "Case=Acc|InflClass=X|Verbform=Fin"


class MorphBackend(Backend):
    """Predicts the same features and UPOS NOUN for every token."""

    def predict(self, sentences):
        vocab = Vocab()
        return [Doc(vocab, words=w, pos=["NOUN"] * len(w), morphs=[PREDICTED] * len(w)) for w in sentences]

    def label_inventory(self, layer):
        return {"Case", "InflClass", "Verbform"} if layer == "morph" else None


class NoInventoryBackend(MorphBackend):
    def label_inventory(self, layer):
        return None


@pytest.fixture(autouse=True)
def morph_backends(monkeypatch):
    monkeypatch.setitem(registry.BUILTINS["backends"], "morph", "seflag.tests.test_morphology:MorphBackend")
    monkeypatch.setitem(registry.BUILTINS["backends"], "noinv", "seflag.tests.test_morphology:NoInventoryBackend")


def evaluate(conllu_dir, backend="morph", **task):
    # one file: gold features Arma=Case=Acc|Number=Plur, virum=Case=Acc, all 5 other words without features
    return run(load_config({
        "language": "la", "model": {"backend": backend}, "output": {"dir": None},
        "task": {"type": "morphology", **task},
        "dataset": {"source": "conllu", "path": str(conllu_dir / "la_test-ud-test.conllu")},
    }))


def test_universal_features_with_renaming(conllu_dir):
    result = evaluate(conllu_dir, predicted_feature_names={"Verbform": "VerbForm"})
    m = result.metrics
    assert m["scored"] == 7
    assert m["accuracy"] == 0  # every word gets an extra VerbForm=Fin
    assert m["pair_precision"] == pytest.approx(2 / 14)
    assert m["pair_recall"] == pytest.approx(2 / 3)
    assert m["per_feature"]["Case"] == {"precision": 2 / 7, "recall": 1.0, "f1": pytest.approx(4 / 9),
                                        "support": 2, "predicted": 7}
    assert "InflClass" not in m["per_feature"]  # not a universal feature
    assert result.details["features_never_predicted"] == ["Number"]
    assert result.details["features_not_in_gold"] == ["VerbForm"]
    assert {"feature": "VerbForm", "gold": "_", "predicted": "Fin", "count": 7} in result.details["top_feature_errors"]


def test_without_renaming_unknown_feature_names_are_dropped(conllu_dir):
    result = evaluate(conllu_dir, with_upos=True)
    assert result.metrics["accuracy"] == pytest.approx(1 / 7)  # only 'virum' (Case=Acc) matches exactly
    assert result.metrics["upos_feats_accuracy"] == pytest.approx(1 / 7)


def test_explicit_all_and_value_mapping(conllu_dir):
    assert evaluate(conllu_dir, features=["Number"]).metrics["accuracy"] == pytest.approx(6 / 7)
    assert "InflClass" in evaluate(conllu_dir, features="all").metrics["per_feature"]
    mapped = evaluate(conllu_dir, features=["Case"], predicted_feature_values={"Case": {"Acc": "Nom"}})
    assert mapped.metrics["pair_precision"] == 0


def test_model_inventory(conllu_dir):
    result = evaluate(conllu_dir, features="model", predicted_feature_names={"Verbform": "VerbForm"})
    assert result.details["evaluated_features"] == ["Case", "VerbForm"]
    assert set(result.metrics["per_feature"]) == {"Case", "VerbForm"}  # Number not evaluated
    with pytest.raises(ConfigError, match="inventory"):
        evaluate(conllu_dir, backend="noinv", features="model")


def test_by_subset(conllu_dir):
    result = run(load_config({
        "language": "la", "model": {"backend": "morph"}, "output": {"dir": None}, "task": "morphology",
        "dataset": {"source": "conllu", "path": str(conllu_dir)}}))
    assert set(result.by_subset) == {"la_test-ud-test", "la_other-ud-dev"}
    assert result.by_subset["la_test-ud-test"]["scored"] == 7
