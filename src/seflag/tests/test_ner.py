import pytest
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab

from seflag import load_config, registry, run
from seflag.backends.base import Backend
from seflag.data.records import EntitySpan, SentenceRecord, TokenRecord, record_to_doc
from seflag.data.tags import TagScheme, decode
from seflag.errors import ConfigError, DataError

E = EntitySpan
PREFIX, SUFFIX = TagScheme(), TagScheme(position="suffix")


@pytest.mark.parametrize("tags, scheme, expected", [
    (["B-PER", "I-PER", "O", "B-LOC"], PREFIX, [E(1, 2, "PER"), E(4, 4, "LOC")]),
    (["B-PER", "B-PER"], PREFIX, [E(1, 1, "PER"), E(2, 2, "PER")]),
    (["O", "I-PER", "I-PER"], PREFIX, [E(2, 3, "PER")]),  # IOB1 / conlleval: I without B starts an entity
    (["B-PER", "I-LOC"], PREFIX, [E(1, 1, "PER"), E(2, 2, "LOC")]),  # type change starts a new entity
    (["S-PER", "B-LOC", "E-LOC", "I-LOC"], PREFIX, [E(1, 1, "PER"), E(2, 3, "LOC"), E(4, 4, "LOC")]),  # BIOES
    (["U-PER", "B-LOC", "L-LOC"], PREFIX, [E(1, 1, "PER"), E(2, 3, "LOC")]),  # BILOU
    (["PER", "PER", "O", "LOC"], PREFIX, [E(1, 2, "PER"), E(4, 4, "LOC")]),  # IO without markers
    (["PRS-B", "PRS-I", "0", "GEO-B"], SUFFIX, [E(1, 2, "PRS"), E(4, 4, "GEO")]),  # Herodotos
    (["B-NORP-X", "I-NORP-X"], PREFIX, [E(1, 2, "NORP-X")]),  # separator inside the type
])
def test_decode(tags, scheme, expected):
    assert decode(tags, scheme) == expected


def test_overlapping_gold_entities_are_rejected():
    record = SentenceRecord(tokens=[TokenRecord(form=w) for w in "a b c".split()],
                            entities=[E(1, 2, "X"), E(2, 3, "Y")])
    with pytest.raises(DataError, match="overlapping"):
        record_to_doc(Vocab(), record, {"entities"})


# --- columns reader ---------------------------------------------------------------------------------------

HERODOTOS = "Label\tWord\nPRS-B\tC.\nPRS-I\tIulius\n0\tvenit\n0\t.\n\nGEO-B\tRomam\n0\tvenit\n"


def columns_dataset(tmp_path, content, **params):
    from seflag.data.columns import ColumnsDataset

    (tmp_path / "d.tsv").write_text(content, encoding="utf-8")
    config = load_config({"language": "la", "task": "ner", "model": {"backend": "spacy", "name": "x"},
                          "dataset": {"source": "columns", "path": str(tmp_path / "d.tsv")}})
    return ColumnsDataset(ColumnsDataset.Config(path=str(tmp_path / "d.tsv"), **params), config)


def test_columns_reader(tmp_path):
    ds = columns_dataset(tmp_path, HERODOTOS, columns={"form": "Word", "ner": "Label"},
                         tag_scheme={"position": "suffix"})
    first, second = ds.records()
    assert [t.form for t in first.tokens] == ["C.", "Iulius", "venit", "."]
    assert first.entities == [E(1, 2, "PRS")] and second.entities == [E(1, 1, "GEO")]
    assert (first.sent_id, first.subset) == ("d-1", "d")


def test_columns_by_index_and_split_after(tmp_path):
    ds = columns_dataset(tmp_path, "Ἀθῆναι\tB-LOC\nεἰσιν\tO\n.\tO\nκαλαί\tO\n", header=False,
                         columns={"form": 0, "ner": 1}, split_after=["."])
    records = list(ds.records())
    assert [len(r.tokens) for r in records] == [3, 1]
    assert records[0].entities == [E(1, 1, "LOC")]


def test_columns_errors(tmp_path):
    with pytest.raises(DataError, match="no column 'Tag'"):
        list(columns_dataset(tmp_path, HERODOTOS, columns={"form": "Word", "ner": "Tag"}).records())
    with pytest.raises(DataError, match=r"d.tsv:3"):
        list(columns_dataset(tmp_path, "w\tt\na\tO\nb\n", columns={"form": "w", "ner": "t"}).records())
    with pytest.raises(ConfigError):
        columns_dataset(tmp_path, HERODOTOS, columns={"ner": "Label"})


# --- task ---------------------------------------------------------------------------------------------------

GOLD = ("Label\tWord\nPRS-B\tGaius\nPRS-I\tIulius\n0\tvidit\nGEO-B\tRomam\n0\tet\nGRP-B\tGallos\n0\tet\n"
        "PRS-B\tBrutum\n")
# predictions: Gaius Iulius = PERSON (correct), Romam = PERSON (wrong label), Gallos missed,
# Brutum = PERSON (correct), 'vidit' = LOC (spurious), 'et Brutum' would be a boundary error (not used here)
PRED = [(0, 2, "PERSON"), (2, 3, "LOC"), (3, 4, "PERSON"), (7, 8, "PERSON")]


class ScriptedNer(Backend):
    entities = PRED

    def predict(self, sentences):
        docs = []
        for words in sentences:
            doc = Doc(Vocab(), words=words)
            doc.ents = [Span(doc, s, e, label=lab) for s, e, lab in type(self).entities]
            docs.append(doc)
        return docs

    def label_inventory(self, layer):
        return {"PERSON", "LOC", "NORP"} if layer == "entities" else None


@pytest.fixture
def evaluate(tmp_path, monkeypatch):
    monkeypatch.setitem(registry.BUILTINS["backends"], "scripted", "seflag.tests.test_ner:ScriptedNer")
    (tmp_path / "gold.tsv").write_text(GOLD, encoding="utf-8")

    def _evaluate(entities=PRED, **task):
        monkeypatch.setattr(ScriptedNer, "entities", entities)
        return run(load_config({
            "language": "la", "model": {"backend": "scripted"}, "output": {"dir": None},
            "task": {"type": "ner", "gold_labels": {"PRS": "person", "GEO": "location", "GRP": "group"},
                     "predicted_labels": {"PERSON": "person", "LOC": "location", "NORP": "group"}, **task},
            "dataset": {"source": "columns", "path": str(tmp_path / "gold.tsv"),
                        "columns": {"form": "Word", "ner": "Label"}, "tag_scheme": {"position": "suffix"}}}))
    return _evaluate


def test_entity_level_scores(evaluate):
    result = evaluate()
    m = result.metrics
    assert (m["gold_entities"], m["predicted_entities"], m["correct_entities"]) == (4, 4, 2)
    assert m["precision"] == m["recall"] == m["f1"] == 0.5
    assert m["unlabeled"]["f1"] == pytest.approx(3 / 4)  # Romam: right span, wrong label
    assert m["per_label"]["person"] == {"precision": 2 / 3, "recall": 1.0, "f1": 0.8, "support": 2, "predicted": 3}
    assert m["per_label"]["group"]["recall"] == 0
    assert m["token_level"]["accuracy"] == pytest.approx(5 / 8)  # vidit, Romam, Gallos wrong
    assert result.details["error_types"] == {"wrong label": 1, "missed": 1, "spurious": 1}
    assert result.details["model_labels"] == ["LOC", "NORP", "PERSON"]
    assert result.details["gold_labels_raw"] == {"PRS": 2, "GEO": 1, "GRP": 1}


def test_boundary_errors_and_label_selection(evaluate):
    boundary = [(0, 1, "PERSON")]  # only 'Gaius' instead of 'Gaius Iulius'
    result = evaluate(boundary, labels=["person"])
    assert result.metrics["gold_entities"] == 2  # only persons evaluated
    assert result.details["error_types"] == {"wrong boundaries": 1, "missed": 1}
    assert result.metrics["unlabeled_f1"] == 0


def test_null_mapping_drops_entities(evaluate):
    m = evaluate(predicted_labels={"PERSON": "person", "LOC": None}).metrics
    assert m["predicted_entities"] == 3
    assert m["precision"] == pytest.approx(2 / 3)


def test_columns_from_hub_repository(tmp_path, monkeypatch):
    import huggingface_hub

    from seflag.data.columns import ColumnsDataset

    snapshot = tmp_path / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "test.tsv").write_text("Ἀθῆναι B-LOC B-LOC\nεἰσιν O O\n\nκαλαί O O\n", encoding="utf-8")
    calls = []

    def fake_download(repo, **kwargs):
        calls.append((repo, kwargs))
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    config = load_config({"language": "grc", "task": "ner", "model": {"backend": "spacy", "name": "x"},
                          "dataset": {"source": "columns", "repo": "someone/model"}})
    ds = ColumnsDataset(ColumnsDataset.Config(repo="someone/model", repo_type="model", revision="abc123",
                                              path="test.tsv", header=False, delimiter=" ",
                                              columns={"form": 0, "ner": 2}), config)
    records = list(ds.records())
    assert [len(r.tokens) for r in records] == [2, 1] and records[0].entities == [E(1, 1, "LOC")]
    # only the requested files are downloaded, from the right repository type and revision
    assert calls[0] == ("someone/model", {"repo_type": "model", "revision": "abc123",
                                          "allow_patterns": ["test.tsv", "test.tsv/*"]})
    fingerprint = ds.fingerprint()
    assert fingerprint["resolved_revision"] == "abc123" and list(fingerprint["files"]) == ["test.tsv"]
