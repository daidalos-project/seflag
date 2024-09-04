# SEFLAG

**S**ystematic **E**valuation **F**ramework for NLP models and datasets in **L**atin and **A**ncient **G**reek

## Evaluation Results

### Lemmatization on UD test data

#### Ancient Greek: greCy (grc_proiel_trf)

{'accuracy': 0.8942049121548943}

#### Ancient Greek: greCy (grc_proiel_lg)
{'accuracy': 0.9055216923628541}

#### Latin: LatinCy (la_core_web_lg)

{'accuracy': 0.8843245653143111}

### NER with average metrics (weighted -- micro -- macro)

|               |                    |                |
|---------------|--------------------|----------------|
| **Model**     | flair_grc_bert_ner | LatinCy        |
| **Dataset**   | Yousef et al.      | Herodotos      |
| **Precision** | 96 -- 94 -- 66     | 96 -- 96 -- 59 |
| **Recall**    | 94 -- 94 -- 76     | 96 -- 96 -- 60 |
| **F1**        | 95 -- 94 -- 64     | 96 -- 96 -- 58 |
