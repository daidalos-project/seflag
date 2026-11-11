import pytest
from spacy.tokens import Doc
from spacy.vocab import Vocab

from seflag import load_config, registry, run
from seflag.backends.base import Backend
from seflag.errors import DataError
from seflag.tasks.dependency_parsing import tree_problems

GOLD = """# sent_id = d1
# text = Puer in venit urbem .
1\tPuer\tpuer\tNOUN\t_\tCase=Nom|Number=Sing\t3\tnsubj\t_\t_
2\tin\tin\tADP\t_\t_\t4\tcase\t_\t_
3\tvenit\t_\tVERB\t_\tMood=Ind|Tense=Pres\t0\troot\t_\t_
4\turbem\turbs\tNOUN\t_\tCase=Acc\t3\tobl:lmod\t_\t_
5\t.\t.\tPUNCT\t_\t_\t3\tpunct\t_\t_

"""
# (head, deprel, upos, feats, lemma) per word; differences to gold are marked
BASE = [
    (3, "nsubj", "NOUN", "Case=Nom|Number=Sing", "puerus"),  # lemma wrong -> BLEX error
    (4, "case", "ADP", "", "in"),
    (0, "root", "VERB", "Mood=Ind|Tense=Pres", "veni"),  # gold lemma '_' -> not compared
    (3, "obl", "NOUN", "Case=Gen", "urbs"),  # subtype missing; feats wrong -> MLAS error
    (4, "punct", "PUNCT", "", "."),  # head wrong
]


class ScriptedBackend(Backend):
    words: list[tuple] = BASE

    def predict(self, sentences):
        rows = type(self).words
        return [Doc(Vocab(), words=w, heads=[i if r[0] == 0 else r[0] - 1 for i, r in enumerate(rows)],
                    deps=[r[1] for r in rows], pos=[r[2] for r in rows], morphs=[r[3] for r in rows],
                    lemmas=[r[4] for r in rows]) for w in sentences]


@pytest.fixture
def evaluate(tmp_path, monkeypatch):
    monkeypatch.setitem(registry.BUILTINS["backends"], "scripted", "seflag.tests.test_dependency_parsing:ScriptedBackend")
    (tmp_path / "gold.conllu").write_text(GOLD, encoding="utf-8")

    def _evaluate(words=BASE, **task):
        monkeypatch.setattr(ScriptedBackend, "words", words)
        return run(load_config({"language": "la", "model": {"backend": "scripted"}, "output": {"dir": None},
                                "task": {"type": "dependency_parsing", **task},
                                "dataset": {"source": "conllu", "path": str(tmp_path / "gold.conllu")}}))
    return _evaluate


def test_conll18_metrics(evaluate):
    m = evaluate().metrics
    assert m["uas"] == pytest.approx(4 / 5)
    assert m["las"] == pytest.approx(4 / 5)  # obl vs obl:lmod counts as correct (subtypes ignored)
    assert m["clas"] == 1.0  # content words: Puer, venit, urbem
    assert m["mlas"] == pytest.approx(2 / 3)  # urbem: wrong features
    assert m["blex"] == pytest.approx(2 / 3)  # Puer: wrong lemma; venit: gold lemma '_' not compared
    assert m["per_deprel"]["punct"]["uas"] == 0
    assert m["sentences"] == 1 and m["predicted_multiple_roots"] == 0 and m["predicted_cycles"] == 0


def test_keep_subtypes(evaluate):
    m = evaluate(deprel_subtypes="keep").metrics
    assert m["las"] == pytest.approx(3 / 5)
    assert m["clas"] == pytest.approx(2 / 3)


def test_mlas_checks_functional_children(evaluate):
    words = list(BASE)
    words[3] = (3, "obl", "NOUN", "Case=Acc", "urbs")  # urbem itself now fully correct ...
    words[1] = (4, "case", "ADV", "", "in")  # ... but its 'case' child has the wrong UPOS
    m = evaluate(words).metrics
    assert m["mlas"] == pytest.approx(2 / 3)
    words[1] = (4, "case", "ADP", "", "in")
    assert evaluate(words).metrics["mlas"] == 1.0


def test_clas_precision_uses_system_content_words(evaluate):
    words = list(BASE)
    words[4] = (3, "obl", "PUNCT", "", ".")  # punctuation labelled as a content relation
    detail = evaluate(words).metrics["clas_detail"]
    assert detail["recall"] == 1.0
    assert detail["precision"] == pytest.approx(3 / 4)
    assert detail["f1"] == pytest.approx(6 / 7)


def test_exclude_deprels_and_mapping(evaluate):
    m = evaluate(exclude_deprels=["punct"]).metrics
    assert m["scored"] == 4 and m["excluded"] == 1 and m["uas"] == 1.0
    words = list(BASE)
    words[0] = (3, "SUBJ", "NOUN", "Case=Nom|Number=Sing", "puer")
    assert evaluate(words).metrics["las"] == pytest.approx(3 / 5)
    assert evaluate(words, predicted_deprels={"SUBJ": "nsubj"}).metrics["las"] == pytest.approx(4 / 5)


def test_invalid_predicted_trees(evaluate):
    two_roots = list(BASE)
    two_roots[0] = (0, "root", "NOUN", "", "puer")
    m = evaluate(two_roots).metrics
    assert m["predicted_multiple_roots"] == 1
    with pytest.raises(DataError, match="2 roots"):
        evaluate(two_roots, require_valid_trees=True)
    cycle = list(BASE)
    cycle[1] = (4, "case", "ADP", "", "in")
    cycle[3] = (2, "obl", "NOUN", "Case=Acc", "urbs")
    assert evaluate(cycle).metrics["predicted_cycles"] == 1


def test_tree_problems():
    assert tree_problems([3, 3, 0]) == (1, False)
    assert tree_problems([2, 1, 0]) == (1, True)
    assert tree_problems([0, 0]) == (2, False)
    assert tree_problems([2, 3, 1]) == (0, True)


def test_metric_selection_reduces_required_layers(evaluate, tmp_path):
    # a dataset with only heads and relations can still be evaluated for UAS/LAS
    (tmp_path / "gold.conllu").write_text(
        "1\tA\t_\t_\t_\t_\t0\troot\t_\t_\n2\tB\t_\t_\t_\t_\t1\tobj\t_\t_\n\n", encoding="utf-8")
    words = [(0, "root", "X", "", "a"), (1, "obj", "X", "", "b")]
    m = evaluate(words, metrics=["uas", "las"]).metrics
    assert m["las"] == 1.0 and "mlas" not in m
    with pytest.raises(DataError, match="no 'lemma' annotations"):  # BLEX needs lemmas (checked first)
        evaluate(words)
