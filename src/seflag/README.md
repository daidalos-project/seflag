# seflag

**S**ystematic **E**valuation **F**ramework for NLP models and datasets in **L**atin and **A**ncient **G**reek.

You describe an evaluation in a YAML file: task, language, dataset, model, and how predictions are compared with
the gold standard. SEFLAG runs it without you having to deal with the formats and APIs of the individual NLP
platforms. Every result is saved together with everything needed to reproduce it: the resolved configuration,
package and model versions, and a fingerprint of the data.

* **Tasks:** lemmatization, POS tagging, morphological features, dependency parsing (CoNLL 2018 metrics), named
  entity recognition, and word embeddings (similarity, neighbour retrieval, synonym selection).
* **Models:** spaCy pipelines (e.g. LatinCy, greCy), Hugging Face transformers, Flair, the Morpheus morphological
  analyser, and fastText/word2vec embeddings via gensim.
* **Data:** CoNLL-U and CoNLL-style column files, local or in any Hugging Face Hub repository, Hub datasets, and
  files downloaded by URL with checksum verification (e.g. from Zenodo).

## Installation

```bash
pip install seflag                 # core: spaCy-based models on local CoNLL-U files
pip install "seflag[hf]"           # + datasets from the Hugging Face Hub (needed by most examples)
pip install "seflag[all]"          # everything below
```

Extras: `hf` (Hub datasets), `transformers`, `flair`, `morpheus` (Ancient Greek beta code conversion),
`embeddings` (gensim). Python 3.10–3.13.

Models are installed separately. If a model is missing, `seflag run` prints the exact command, e.g. for LatinCy
and greCy:

```bash
pip install https://huggingface.co/latincy/la_core_web_lg/resolve/eef57207424f80d2e3806315bdc08554f432f740/la_core_web_lg-3.9.8-py3-none-any.whl
pip install https://github.com/jmyerston/greCy/releases/download/v3.7.5/grc_proiel_lg-3.7.5-py3-none-any.whl
```

The URLs are pinned to a specific model version, because pip only installs wheels with a version in their file
name and the model authors replace or delete older wheels. Both models need spaCy 3.8.

Morpheus is downloaded and compiled automatically on first use (Linux and macOS; needs a C compiler and `make`),
unless `$MORPHEUS_PATH` points to an existing installation.

## Platform support

SEFLAG runs on Linux, macOS and Windows (tested in CI with Python 3.10 and 3.13 on Windows and macOS). One
exception: **Morpheus is not available on native Windows.** It is built from C sources with `make`, and
SEFLAG's Docker launcher cannot share its temporary files with a Linux container from Windows. Run SEFLAG in
[WSL](https://learn.microsoft.com/windows/wsl/) instead, where everything works as on Linux. The optional
extras (`hf`, `transformers`, `flair`, `embeddings`) are tested on Linux only.

## Quick start

`latincy_lemma.yaml`:

```yaml
name: latincy-lg-lemma-ud-latin
language: la
task: lemmatization
dataset:
  source: huggingface
  repo: daidalos-project/latin_treebanks_ud_test
  revision: 14ab3708f8f55eaa700ac5828cb3351fe5346474   # pinned for reproducibility
  split: test
model:
  backend: spacy
  name: la_core_web_lg
options:
  label_normalization: [v_to_u, j_to_i]   # the treebanks differ in u/v and i/j spelling; don't count it as an error
```

```bash
seflag run latincy_lemma.yaml     # prints the scores, writes seflag_results/<name>_<timestamp>.json
```

From Python:

```python
from seflag import load_config, run

result = run(load_config("latincy_lemma.yaml", options={"max_sentences": 100}))
print(result.metrics["accuracy"], result.by_subset)
```

Relative paths of datasets and models are resolved against the directory of the configuration file; results are
written to `output.dir` (default `seflag_results`) relative to the current working directory.

## Example configurations

The package ships about 20 ready-to-run configurations for the evaluations behind SEFLAG's published results
(LatinCy, greCy, Morpheus, UGARIT NER models, fastText embeddings). They are installed with the package:

```bash
python -c "import importlib.resources as r; print(r.files('seflag') / 'examples')"
seflag run "$(python -c "import importlib.resources as r; print(r.files('seflag') / 'examples')")/latincy_lemma_ud.yaml"
```

## Comparing several models and datasets

A configuration with a `matrix` section runs every model on every dataset and prints a Markdown table, e.g. for a
README:

```yaml
name: lemmatization-ud
task: lemmatization
matrix:
  datasets:
    - label: UD Latin
      language: la
      dataset: {source: huggingface, repo: daidalos-project/latin_treebanks_ud_test, split: test}
    - label: UD Ancient Greek
      language: grc
      dataset: {source: huggingface, repo: daidalos-project/ancient_greek_treebanks_ud_test, split: test}
  models:
    - label: LatinCy (la_core_web_lg)
      language: la                    # not run on the Greek data
      model: {backend: spacy, name: la_core_web_lg}
    - label: greCy (grc_proiel_lg)
      language: grc
      model: {backend: spacy, name: grc_proiel_lg}
    - label: Morpheus                 # no language: runs on both
      model: {backend: morpheus}
      options: {label_normalization: [strip_homonym_digits]}
  table: {percent: true, digits: 2}
```

The full configuration is `examples/matrix_lemma_ud.yaml` (with pinned revisions and the pipeline components used);
it produces:

| Model | UD Latin | UD Ancient Greek |
|:--|--:|--:|
| LatinCy (la_core_web_lg) | 91.96 | – |
| greCy (grc_proiel_lg) | – | 90.47 |
| Morpheus | 69.12 | 87.55 |

These are the scores of LatinCy 3.9.8 and greCy 3.7.5 (installed with the commands above, spaCy 3.8). SEFLAG v1.0.0
reported results for LatinCy 3.7.4, which is no longer distributed.

Each cell is a complete evaluation with its own result file; a cell that fails (e.g. a model that is not installed)
is reported in the table instead of stopping the run.

## Explicit evaluation decisions

Choices that silently change results are named options and stored with every result, e.g.:

* `options.input_normalization` / `options.label_normalization`: Unicode forms (`nfc`), Latin `v_to_u` and `j_to_i`,
  `lowercase`, Greek elision marks (`elision_to_u02bc` for greCy), `final_sigma`, `lunate_sigma`,
  `strip_homonym_digits` for Morpheus lemmas;
* `options.sentence_segmentation`: gold sentences (default) or the model's own;
* `task.exclude_upos`, `task.features`, `task.deprel_subtypes`, `task.gold_labels` / `task.predicted_labels` (label
  mappings for NER), and more.

The two normalization options act at different points and are independent. `input_normalization` changes the
words the model sees; `label_normalization` changes only what counts as a match, on both the gold and the predicted
side. When the data and the model follow different spelling conventions, you usually need both. Suppose a treebank
writes *venit* with the lemma *venio*, and the model was trained on text with only *u*:

* with only `input_normalization: [v_to_u]`, the model receives *uenit* and predicts *uenio*, which still differs
  from the gold *venio*;
* with only `label_normalization: [v_to_u]`, *uenio* and *venio* match, but the model was given *venit*, a spelling
  it may not know, and may predict a worse lemma.

Other cases need just one of them: Morpheus lemmas need `strip_homonym_digits` only as a label normalization
(*sum1* → *sum*), and the shipped examples normalize labels only, so that the models are tested on the spelling of
the data as it is.

`seflag list -v` shows every task, data source, backend and normalizer with all its options and their allowed
values. Configurations are validated completely when they are loaded, including each component's options, and
misspelled names get a suggestion (`unknown task 'lemmatisation'; did you mean 'lemmatization'?`).

Datasets and models differ in how they encode the same text in Unicode (elision marks, u/v, case, NFC vs NFD), which
can change scores by several points. [ENCODING.md](https://github.com/daidalos-project/seflag/blob/main/src/seflag/ENCODING.md)
documents the conventions of all datasets and models used in the examples, and `seflag inspect-encoding` measures
them for any configuration.

## Commands

| Command | Purpose |
|---|---|
| `seflag run <config>...` | Run evaluations or matrices (`--max-sentences`, `--output`, `--no-save`, `--fail-fast`) |
| `seflag list [-v]` | Available tasks, data sources, backends, normalizers and their options |
| `seflag table <results>...` | Markdown table (models × datasets) from saved result files or directories |
| `seflag inspect-encoding <config>...` | Unicode conventions of a configuration's data and embedding vocabulary |
| `seflag convert-conllu <files> --output <dir>` | Convert CoNLL-U files into SEFLAG's Parquet layout for the Hub |
| `seflag export <config> --output <dir>` | Export any configured dataset to that layout |
| `seflag cache info` / `purge` | Show or delete downloaded files and the Morpheus build |

## Models in separate environments

Backends whose dependencies conflict with others can run in their own environment, created by
[uv](https://docs.astral.sh/uv/) on first use:

```yaml
model:
  backend: flair
  name: UGARIT/flair_grc_bert_ner
  launcher:
    requirements: [flair==0.15.1]
```

## Python API

In Python, configurations are best built from enums and the components' typed `Config` classes instead of strings
and dictionaries: your editor completes the allowed values, and typos fail immediately. Both forms are validated
the same way and produce identical result files.

```python
import seflag
from seflag import Language, Normalizer, Upos
from seflag.backends.spacy import SpacyBackend
from seflag.config import EvaluationOptions
from seflag.data.huggingface import HuggingFaceDataset
from seflag.tasks.lemmatization import LemmatizationTask

config = seflag.load_config({
    "language": Language.LATIN,
    "task": LemmatizationTask.Config(exclude_upos=[Upos.PUNCT]),
    "dataset": HuggingFaceDataset.Config(repo="daidalos-project/latin_treebanks_ud_test",
                                         revision="14ab3708f8f55eaa700ac5828cb3351fe5346474", split="test"),
    "model": SpacyBackend.Config(name="la_core_web_lg"),
    "options": EvaluationOptions(label_normalization=[Normalizer.V_TO_U, Normalizer.J_TO_I]),
})
result = seflag.run(config)
```

All fixed vocabularies (languages, tasks, normalizers, UPOS tags, component options) are enums in `seflag.enums`.
Names of tasks, data sources and backends can be extended by plugins, so `TaskType`, `DatasetSource` and
`BackendName` list the built-in ones; any other name must be registered. Vocabularies that come from the data
(split names, dependency relations, feature names) remain strings.

Beyond running evaluations, the package gives direct access to its data loading, models and results:

```python
from pathlib import Path

import seflag
from seflag.data.huggingface import HuggingFaceDataset

# SEFLAG's data access on its own: a configuration file, or just its dataset section
dataset = seflag.load_dataset(HuggingFaceDataset.Config(repo="daidalos-project/latin_treebanks_ud_test",
                                                        revision="14ab3708f8f55eaa700ac5828cb3351fe5346474",
                                                        split="test"),
                              language=seflag.Language.LATIN)
docs = dataset.docs()               # gold sentences as spaCy Docs, with every annotation layer of the data
records = list(dataset.records())   # or as SentenceRecord objects (CoNLL-U semantics)

model = seflag.load_backend("latincy_lemma.yaml")
predicted = model.predict([["Gallia", "est", "omnis", "divisa", "in", "partes", "tres"]])

# saved results, e.g. to compare runs or build a table for a paper
results = [seflag.EvaluationResult.load(p) for p in sorted(Path("seflag_results").glob("*.json"))]
print(seflag.table_from_results(results).table())
```

## Extending SEFLAG

Datasets, models and tasks are components with the same structure: a class with a nested pydantic `Config` (its
options in the configuration file), a `description`, and one main method. A new model, for example, wraps any NLP
tool behind `predict()`:

```python
from pydantic import BaseModel, ConfigDict
from spacy.tokens import Doc
from spacy.vocab import Vocab

import seflag
from seflag import ComponentKind, Language
from seflag.backends.base import Backend
from seflag.data.conllu import ConlluDataset
from seflag.tasks.lemmatization import LemmatizationTask


class LowercaseLemmatizer(Backend):
    """Baseline: the lemma of a word is the word in lower case."""

    description = "Lower-case baseline lemmatizer"

    class Config(BaseModel):            # options of the configuration's model section
        model_config = ConfigDict(extra="forbid")
        keep_proper_names: bool = False

    def predict(self, sentences: list[list[str]]) -> list[Doc]:
        vocab = Vocab()
        lemmas = [[w if self.config.keep_proper_names and w[:1].isupper() else w.lower() for w in words]
                  for words in sentences]
        return [Doc(vocab, words=words, lemmas=lem) for words, lem in zip(sentences, lemmas, strict=True)]

    def metadata(self) -> dict:          # stored in the provenance of every result
        return {"name": "lowercase-baseline", "keep_proper_names": self.config.keep_proper_names}


seflag.register(ComponentKind.BACKENDS, "lowercase", LowercaseLemmatizer)
result = seflag.run(seflag.load_config({
    "language": Language.LATIN,
    "task": LemmatizationTask.Config(),
    "dataset": ConlluDataset.Config(path="my_treebank/"),
    "model": LowercaseLemmatizer.Config(keep_proper_names=True),
}))
print(result.metrics["accuracy"])
```

The rules for each kind of component:

* **Models** (`seflag.backends.base.Backend`) receive the gold tokens and must return one `Doc` per sentence with
  exactly these tokens; SEFLAG checks this after every prediction (`check_predictions`) and reports the first
  difference. Optional: `label_inventory()` (labels the model can predict), `metadata()` (versions for the
  provenance). Word embeddings implement `seflag.backends.embeddings.EmbeddingBackend` instead.
* **Datasets** (`seflag.data.base.DatasetAdapter`) yield `seflag.data.records.SentenceRecord` objects from
  `records()` and identify the exact data version in `fingerprint()`.
* **Tasks** (`seflag.tasks.base.Task`) declare the gold annotation layers they need (`required_layers`) and compare
  gold with predicted Docs in `evaluate()`, returning a `TaskResult`; `primary_metric` names the score shown in
  tables.

`seflag.register()` makes a component available in the current Python session. To share it, put it in a package
and declare an entry point; SEFLAG then finds it without any registration (also in separate environments started
by a `launcher`):

```toml
[project.entry-points."seflag.backends"]
lowercase = "my_package.lemmatizers:LowercaseLemmatizer"
```

The built-in components in `seflag/backends/`, `seflag/data/` and `seflag/tasks/` are complete examples, and
`seflag list -v` shows how SEFLAG sees a registered component.

## Data and licenses

SEFLAG contains no datasets or models. They are downloaded on first use from their repositories, pinned to
revisions or verified by checksums, and keep their own licenses. Several of the UD treebanks used in the examples,
for instance, are licensed CC BY-NC-SA. The `daidalos-project` organisation on the Hugging Face Hub mirrors the
aggregated UD test sets for Latin and Ancient Greek, with the licenses of the individual treebanks.

The seflag package itself is licensed under the [Zero-Clause BSD license](https://opensource.org/license/0bsd).

## Citation

Please cite SEFLAG via its Zenodo record: [doi:10.5281/zenodo.13951323](https://doi.org/10.5281/zenodo.13951323)
(always the latest version; each release also has its own DOI there). Package, documentation and published
evaluation results share one version number: v1.0.0 is the release described in
[Schulz & Deichsler (2024)](https://aclanthology.org/2024.nlp4dh-1.24/), and the PyPI package starts with v1.1.0.
