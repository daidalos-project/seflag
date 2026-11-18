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

Models are installed separately. If a model is missing, `seflag run` prints the exact command, e.g. for LatinCy:

```bash
pip install https://huggingface.co/latincy/la_core_web_lg/resolve/main/la_core_web_lg-any-py3-none-any.whl
```

Morpheus is downloaded and compiled automatically on first use (Linux and macOS; needs a C compiler and `make`),
unless `$MORPHEUS_PATH` points to an existing installation.

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
  label_normalization: [v_to_u, j_to_i]   # LatinCy writes lemmas with u and i; some treebanks use v and j
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
| LatinCy (la_core_web_lg) | 89.05 | – |
| greCy (grc_proiel_lg) | – | 90.56 |
| Morpheus | 69.12 | 86.13 |

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

`seflag list -v` shows every task, data source, backend and normalizer with all its options.

Datasets and models differ in how they encode the same text in Unicode (elision marks, u/v, case, NFC vs NFD), which
can change scores by several points. [ENCODING.md](https://github.com/daidalos-project/seflag/blob/main/src/seflag/ENCODING.md)
documents the conventions of all datasets and models used in the examples, and `seflag inspect-encoding` measures
them for any configuration.

## Commands

| Command | Purpose |
|---|---|
| `seflag run <config>...` | Run evaluations or matrices (`--max-sentences`, `--output`, `--no-save`, `--fail-fast`) |
| `seflag list [-v]` | Available tasks, data sources, backends, normalizers and their options |
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
