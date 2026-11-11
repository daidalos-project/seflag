import hashlib

import pytest

from seflag import load_config, run
from seflag.errors import ConfigError, DataError
from seflag.metrics import pearson, spearman

pytest.importorskip("gensim")

# word2vec text format; order = frequency order (Roma before roma)
VECTORS = "6 2\nrex 1 0\nregina 0.9 0.1\nmiles 0 1\ngladius 0.1 0.9\nRoma 0.7 0.7\nroma 0.6 0.8\n"


@pytest.fixture
def files(tmp_path):
    (tmp_path / "model.vec").write_text(VECTORS, encoding="utf-8")
    (tmp_path / "tie.vec").write_text("3 2\nrex 1 0\nregina 0.9 0.1\nimperator 0.9 0.1\n", encoding="utf-8")
    (tmp_path / "pairs.json").write_text(
        '{"pairs": [{"pair": "rex, regina", "score": 9, "set": "a"}, {"pair": "rex, miles", "score": 1, "set": "a"},'
        ' {"pair": "miles, gladius", "score": 8, "set": "b"}, {"pair": "rex, gladius", "score": 2, "set": "a"},'
        ' {"pair": "rex, ignotus", "score": 5, "set": "a"}]}', encoding="utf-8")
    (tmp_path / "related.tsv").write_text("w1\tw2\nrex\tregina\nmiles\tgladius\nrex\tRoma\n", encoding="utf-8")
    (tmp_path / "choices.tsv").write_text("rex\tregina\tmiles\tgladius\nmiles\tgladius\trex\tignotus\n",
                                          encoding="utf-8")
    return tmp_path


def evaluate(files, task, dataset, model=None, model_file="model.vec"):
    return run(load_config({
        "language": "la", "task": task, "output": {"dir": None}, "dataset": dataset,
        "model": {"backend": "gensim", "path": str(files / model_file), "format": "word2vec_text", **(model or {})}}))


def test_correlations_against_hand_computed_values():
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1)
    # ranks [1, 2, 3, 4] vs [2, 3.5, 3.5, 1] (tie -> average rank)
    assert spearman([1, 2, 3, 4], [10, 20, 20, 5]) == pytest.approx(-1.5 / 22.5 ** 0.5)
    assert spearman([4, 1, 3, 2], [3.5, 1, 3.5, 2]) == pytest.approx(4.5 / 22.5 ** 0.5)
    assert pearson([1], [1]) is None and pearson([1, 1], [1, 2]) is None
    scipy = pytest.importorskip("scipy.stats")
    x, y = [3, 1, 4, 1, 5, 9, 2, 6], [2, 7, 1, 8, 2, 8, 1, 8]
    assert spearman(x, y) == pytest.approx(scipy.spearmanr(x, y).statistic)
    assert pearson(x, y) == pytest.approx(scipy.pearsonr(x, y).statistic)


JSON_PAIRS = {"source": "word_pairs", "format": "json", "records_key": "pairs", "pair_field": "pair",
              "score_field": "score"}


def test_word_similarity(files):
    m = evaluate(files, "word_similarity", {**JSON_PAIRS, "path": str(files / "pairs.json")}).metrics
    # gold [9, 1, 8, 2] vs cosine [0.994, 0, 0.994, 0.110]: ranks [4, 1, 3, 2] vs [3.5, 1, 3.5, 2]
    assert m["spearman"] == pytest.approx(4.5 / 22.5 ** 0.5)
    assert (m["pairs"], m["evaluated_pairs"], m["coverage"]) == (5, 4, 0.8)
    assert m["oov_words"] == 1
    m = evaluate(files, "word_similarity", {**JSON_PAIRS, "path": str(files / "pairs.json"), "where": {"set": "b"}})
    assert m.metrics["pairs"] == 1


def test_word_retrieval_and_vocabulary_options(files):
    dataset = {"source": "word_pairs", "format": "tsv", "header": True, "columns": {"word1": "w1", "word2": "w2"},
               "path": str(files / "related.tsv")}
    m = evaluate(files, {"type": "word_retrieval", "k": [1, 10]}, dataset).metrics
    # rex -> regina (rank 1), miles -> gladius (1), rex -> Roma (2: after regina)
    assert m["hits_at_1"] == pytest.approx(2 / 3) and m["hits_at_10"] == 1
    assert m["mrr_at_10"] == pytest.approx(2.5 / 3)
    # with only the 3 most frequent words, gladius and Roma are unknown
    m = evaluate(files, "word_retrieval", dataset, {"max_vocabulary": 3}).metrics
    assert m["evaluated_queries"] == 1 and m["oov_words"] == 2
    assert evaluate(files, {"type": "word_retrieval", "directions": "both"}, dataset).metrics["queries"] == 6


def test_case_insensitive_lookup(files):
    from seflag.backends.embeddings import GensimBackend

    config = load_config({"language": "la", "task": "word_similarity", "dataset": {"source": "word_pairs"},
                          "model": {"backend": "gensim"}})
    backend = GensimBackend(GensimBackend.Config(path=str(files / "model.vec"), format="word2vec_text",
                                                 case_insensitive=True), config)
    assert backend.key("ROMA") == "Roma"  # the more frequent of 'Roma' and 'roma'
    assert backend.key("Rex") == "rex"
    exact = GensimBackend(GensimBackend.Config(path=str(files / "model.vec"), format="word2vec_text"), config)
    assert exact.key("ROMA") is None and exact.key("roma") == "roma"


def test_synonym_selection(files):
    dataset = {"source": "word_choices", "path": str(files / "choices.tsv")}
    m = evaluate(files, "synonym_selection", dataset).metrics
    assert (m["accuracy"], m["evaluated_items"]) == (1.0, 1)  # second item skipped: 'ignotus' unknown
    m = evaluate(files, {"type": "synonym_selection", "missing_candidates": "ignore"}, dataset).metrics
    assert m["evaluated_items"] == 2 and m["accuracy"] == 1.0
    assert m["chance_accuracy"] == pytest.approx((1 / 3 + 1 / 2) / 2)


def test_ties_count_as_wrong(files):
    (files / "tie.tsv").write_text("rex\tregina\timperator\n", encoding="utf-8")
    m = evaluate(files, "synonym_selection", {"source": "word_choices", "path": str(files / "tie.tsv")},
                 model_file="tie.vec").metrics
    assert m["accuracy"] == 0.0


def test_invalid_combinations(files, conllu_dir):
    with pytest.raises(ConfigError, match="do not fit together"):
        evaluate(files, "lemmatization", {"source": "word_choices", "path": str(files / "choices.tsv")})
    with pytest.raises(ConfigError, match="subwords"):
        evaluate(files, {"type": "synonym_selection", "oov": "subwords"},
                 {"source": "word_choices", "path": str(files / "choices.tsv")})
    with pytest.raises(ConfigError, match="launcher"):
        evaluate(files, "synonym_selection", {"source": "word_choices", "path": str(files / "choices.tsv")},
                 {"launcher": {"command": ["python"]}})
    with pytest.raises(DataError, match="gold scores"):
        evaluate(files, "word_similarity", {"source": "word_pairs", "format": "tsv", "header": True,
                                            "path": str(files / "related.tsv")})


def test_download_with_checksum(tmp_path, monkeypatch):
    from seflag.data.download import download

    monkeypatch.setenv("SEFLAG_HOME", str(tmp_path / "home"))
    source = tmp_path / "pairs.tsv"
    source.write_text("a\tb\t1\n", encoding="utf-8")
    md5 = hashlib.md5(source.read_bytes()).hexdigest()
    path = download(source.as_uri(), f"md5:{md5}")
    assert path.read_bytes() == source.read_bytes() and path.name == "pairs.tsv"
    assert download(source.as_uri(), f"md5:{md5}") == path  # cached
    other = tmp_path / "other.tsv"
    other.write_text("c\td\t2\n", encoding="utf-8")
    with pytest.raises(DataError, match="Checksum mismatch"):
        download(other.as_uri(), "md5:" + "0" * 32)
    assert not any(p.suffix == ".part" for p in (tmp_path / "home").rglob("*"))  # no leftovers
    with pytest.raises(ConfigError, match="Invalid checksum"):
        download(source.as_uri(), "crc:123")
