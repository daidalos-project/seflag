import json
import unicodedata

import yaml

from seflag.cli import main
from seflag.encoding import analyse, normalization_class

GREEK_CONLLU = """# sent_id = g1
1\tἀλλ̓\tἀλλά\tCCONJ\t_\t_\t0\troot\t_\t_
2\tπαῤ\tπαρά\tADP\t_\t_\t1\tcase\t_\t_
3\tλόγοσ\tλόγος\tNOUN\t_\t_\t1\tobj\t_\t_
4\tκovmh\tκώμη\tNOUN\t_\t_\t1\tobj\t_\t_
5\t·\t·\tPUNCT\t_\t_\t1\tpunct\t_\t_

"""


def test_normalization_classes():
    nfc = "λόγος"
    assert [normalization_class(s) for s in ["amo", nfc, unicodedata.normalize("NFD", nfc), "λόγος", "αβ"]] == \
        ["ascii", "nfc", "nfd", "mixed", "no_decomposable"]


def test_analyse_greek_properties():
    stats = analyse(["ἀλλ̓", "παῤ", "λόγοσ", "κovmh", "ϲοφία", "τῇ", "τῆι", "καὶ", "Ἕλλην", "", None])
    assert stats.strings == 9 and stats.empty == 2
    for key in ("elided", "medial_sigma_at_end", "mixed_greek_latin", "lunate_sigma", "iota_subscript",
                "prosgegrammeni", "grave_accent", "capitalized"):
        assert stats.features[key] >= 1, key
    assert stats.features["elided"] == 2
    assert stats.characters["changed_by_nfc"]["ό"] == 1  # oxia
    assert stats.normalization["mixed"] == 2  # oxia word, and U+1FBE (NFC and NFD both replace it)
    lines = "\n".join(stats.format())
    assert "2 not NFC" in lines and "U+1F79 GREEK SMALL LETTER OMICRON WITH OXIA" in lines


def test_analyse_latin_properties():
    stats = analyse(["virum", "Iuppiter", "canō", "sum1", "caelum", "sсriptor", " "])
    assert stats.features["v"] == 1 and stats.features["macron_or_breve"] == 1
    assert stats.features["letters_and_digits"] == 1 and stats.features["other_script"] == 1  # Cyrillic с
    assert stats.characters["invisible"][" "] == 1


def test_inspect_encoding_cli(tmp_path, conllu_dir, capsys):
    (conllu_dir / "grc_test-ud-test.conllu").write_text(GREEK_CONLLU, encoding="utf-8")
    config = tmp_path / "eval.yaml"
    config.write_text(yaml.safe_dump({"language": "grc", "task": "lemmatization", "model": {"backend": "spacy",
                                      "name": "none"}, "dataset": {"source": "conllu", "path": str(conllu_dir)}}))
    assert main(["inspect-encoding", str(config), "--by-subset"]) == 0
    out = capsys.readouterr().out
    assert "[all] form" in out and "[grc_test-ud-test] lemma" in out and "elided: 2" in out
    assert main(["inspect-encoding", str(config), "--json", "--max-sentences", "2"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report[0]["kind"] == "dataset" and report[0]["subsets"][""]["form"]["strings"] == 9
    assert main(["inspect-encoding", str(config), "--model"]) == 0  # spaCy: no vocabulary, only a note
    assert "only available for embedding backends" in capsys.readouterr().err
