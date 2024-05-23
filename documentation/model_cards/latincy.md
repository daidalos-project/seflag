**la_core_web_lg**

- **Person or organization developing model**: [Patrick J. Burns; with
Nora Bernhardt \[ner\], Tim Geelhaar \[tagger, morphologizer, parser,
ner\], Vincent Koch \[ner\]](https://diyclassics.github.io/)

- **Model date**: May 2023

- **Model version: 3.7.4**

- Model type: spaCy

- Information about training algorithms, parameters, fairness
constraints or other applied approaches, and features: For information on the training workflow see p.4-5 of LatinCy: Synthetic Trained Pipelines for Latin NLP
(https://arxiv.org/pdf/2305.04365v1)

- Paper or other resource for more information: **Burns, P.J. 2023.
"LatinCy: Synthetic Trained Pipelines for Latin NLP." arXiv:2305.04365
\[cs.CL\]. http://arxiv.org/abs/2305.04365.

- License: *MIT*

- Where to send questions or comments about the model:
https://diyclassics.github.io/

Intended Use

- Primary intended uses: Morphological analysis, POS-Tagging,
Lemmatizing, Parsing, NER

- Primary intended users: Classical Scholars

- Out-of-scope use cases: unknown

Data, Limitations, and Recommendations

- Data selection for training: Training data consists of latin
UD-Treebanks, Wikipedia and OSCAR sentence data, the CC-100 Latin
dataset and the Herodotos Project NER dataset

- Data selection for evaluation: Evaluation was done according to the
spaCy workflow and is documented in the meta.json file found in the
repository
(https://huggingface.co/latincy/la_core_web_lg/blob/main/meta.json)

- Limitations: unknown
