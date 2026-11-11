import unicodedata

import pytest

from seflag.metrics import accuracy, classification_report, confusion_matrix, top_errors
from seflag.normalization import get_pipeline


def test_pipeline_applies_in_order():
    assert get_pipeline(["v_to_u", "lowercase"])("Venio") == "uenio"
    assert get_pipeline(["strip_homonym_digits"])("sum1") == "sum"
    assert get_pipeline(["strip_homonym_digits"])("1") == "1"
    assert get_pipeline(["strip_homonym_digits"])("qui#1") == "qui"
    assert get_pipeline(["strip_diacritics"])("λόγος") == "λογος"
    decomposed = unicodedata.normalize("NFD", "ἄνθρωπος")
    assert get_pipeline(["nfc"])(decomposed) == "ἄνθρωπος"


def test_unknown_normalizer():
    with pytest.raises(ValueError):
        get_pipeline(["nope"])


def test_metrics():
    gold = ["A", "A", "B", "C"]
    pred = ["A", "B", "B", "D"]
    assert accuracy(gold, pred) == 0.5
    assert accuracy([], []) is None
    report = classification_report(gold, pred)
    assert report["per_label"]["A"] == {"precision": 1.0, "recall": 0.5, "f1": 2 / 3, "support": 2}
    assert report["per_label"]["B"]["precision"] == 0.5
    assert report["per_label"]["D"]["support"] == 0
    assert report["macro_recall"] == pytest.approx((0.5 + 1 + 0 + 0) / 4)
    assert report["weighted_recall"] == pytest.approx(0.5)  # equals accuracy for single-label tasks
    assert confusion_matrix(gold, pred)["A"] == {"A": 1, "B": 1}
    assert top_errors(gold, pred, 1) == [{"gold": "A", "predicted": "B", "count": 1}]
    with pytest.raises(ValueError):
        accuracy(["A"], [])
