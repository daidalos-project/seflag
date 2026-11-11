import pytest
from spacy.vocab import Vocab

from seflag.config import load_config
from seflag.data.conllu import ConlluDataset, parse_conllu_lines, read_conllu_files, write_conllu
from seflag.data.records import (EmptyNode, MultiwordToken, SentenceRecord, TokenRecord, check_layers_present,
                                 record_to_doc)
from seflag.errors import DataError


def make_dataset(path, **params):
    config = load_config({"language": "la", "task": "lemmatization", "dataset": {"source": "conllu", "path": str(path)},
                          "model": {"backend": "spacy", "name": "x"}})
    return ConlluDataset(ConlluDataset.Config(path=str(path), **params), config)


def test_conllu_records_skip_multiword_tokens(conllu_dir):
    records = list(make_dataset(conllu_dir / "la_test-ud-test.conllu").records())
    assert [r.sent_id for r in records] == ["s1", "s2"]
    first = records[0]
    assert [t.form for t in first.tokens] == ["Arma", "virum", "que", "cano"]
    assert first.tokens[1] == TokenRecord(form="virum", lemma="vir", upos="NOUN", feats="Case=Acc", head=4,
                                          deprel="obj")
    assert first.tokens[0].feats == "Case=Acc|Number=Plur"
    assert first.tokens[3].head == 0
    assert first.subset == "la_test-ud-test"
    assert records[1].tokens[2].lemma is None  # '_' = missing
    assert SentenceRecord.from_dict(first.to_dict()) == first


def test_conllu_records_keep_everything(conllu_dir):
    first, second = make_dataset(conllu_dir / "la_test-ud-test.conllu").records()
    assert first.multiword_tokens == [MultiwordToken(first=2, last=3, form="virumque", misc="SpaceAfter=No")]
    assert second.sent_id == "s2" and second.text == "Troiae qui primus"  # comment order differs
    assert second.comments == ["# source = Vergil, Aeneid 1.1", "# text = Troiae qui primus", "# sent_id = s2",
                               "# newpar"]
    assert second.metadata == {"source": "Vergil, Aeneid 1.1", "text": "Troiae qui primus", "sent_id": "s2",
                               "newpar": None}
    assert second.tokens[0].misc == "Gloss=of-Troy"
    assert second.tokens[1].deps == "3:nsubj"
    assert second.empty_nodes == [EmptyNode(id="3.1", form="fuit", lemma="sum", upos="AUX", deps="3:cop")]


def test_conllu_roundtrip_is_byte_identical(conllu_dir, tmp_path):
    source = conllu_dir / "la_test-ud-test.conllu"
    records = list(read_conllu_files([source]))
    write_conllu([SentenceRecord.from_dict(r.to_dict()) for r in records], tmp_path / "out.conllu")
    assert (tmp_path / "out.conllu").read_bytes() == source.read_bytes()


@pytest.mark.parametrize("bad, message", [
    ("1\tA\ta\n\n", "10 tab-separated"),
    ("2\tA\ta\tX\t_\t_\t0\troot\t_\t_\n\n", "out of sequence"),
    ("1\tA\ta\tX\t_\t_\tx\troot\t_\t_\n\n", "invalid HEAD"),
    ("1\tA\ta\tX\t_\t_\t0\troot\t_\t_\n# late comment\n\n", "inside a sentence"),
])
def test_conllu_errors_name_the_line(bad, message):
    with pytest.raises(DataError, match=message):
        list(parse_conllu_lines(bad.splitlines(keepends=True), "x.conllu"))


def test_directory_and_fingerprint(conllu_dir):
    ds = make_dataset(conllu_dir)
    assert len(list(ds.records())) == 4
    assert len(ds.fingerprint()["files"]) == 2


def test_missing_path(tmp_path):
    with pytest.raises(DataError):
        list(make_dataset(tmp_path / "missing").records())


def test_record_to_doc_layers(conllu_dir):
    record = next(make_dataset(conllu_dir / "la_test-ud-test.conllu").records())
    doc = record_to_doc(Vocab(), record, {"lemma", "upos"})
    assert [t.lemma_ for t in doc] == ["arma", "vir", "que", "cano"]
    assert [t.pos_ for t in doc] == ["NOUN", "NOUN", "CCONJ", "VERB"]
    assert doc.user_data["sent_id"] == "s1"
    bad = SentenceRecord(tokens=[TokenRecord(form="x", upos="FOO")])
    with pytest.raises(DataError):
        record_to_doc(Vocab(), bad, {"upos"})


def test_missing_layer_is_reported():
    records = [SentenceRecord(tokens=[TokenRecord(form="a", upos="NOUN")])]
    check_layers_present(records, {"upos"})
    with pytest.raises(DataError, match="'lemma'"):
        check_layers_present(records, {"lemma"})
