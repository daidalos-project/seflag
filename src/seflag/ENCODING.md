# Character encoding of SEFLAG's datasets and models

Status: 2026-11-18. Covers every dataset and model referenced by the configurations in `examples/`.

All of these resources are UTF-8, but that says little about how a word is *spelled in code points*. Unicode
offers several encodings of the same Greek or Latin text: precomposed or decomposed accents (NFC/NFD), oxia or
tonos, six different elision marks, σ or ς at the end of a word, u or v, i or j. A model that receives a spelling
it did not see in training, or whose output is compared with a gold standard spelled differently, loses accuracy.
That loss is caused by the encoding, not by what the model knows about the language. This file records each
resource's conventions, the mismatches between them, how much they cost, and the normalizers SEFLAG provides
against them.

## Summary

| Discrepancy | Where | Measured effect |
|---|---|---|
| Greek elision mark: each treebank uses a different one (U+0313, U+2019, U+1FBF); greCy only recognises U+02BC, Morpheus only U+2019/U+0027, the BERT tokenizer turns U+02BC/U+1FBD/U+1FBF into `[UNK]` | UD Greek, NER (Yousef), all Greek models | greCy lemma accuracy 0.9056 → 0.9226 with `input_normalization: [elision_to_u02bc]`; Morpheus 0.8613 → 0.8755 after the backend fix; 719 NER test tokens (1.8%) become `[UNK]` (no measurable effect on NER) |
| Latin u/v and i/j: LatinCy outputs u/i lemmas, Morpheus v/j lemmas, the UD treebanks use both | UD Latin, LatinCy, Morpheus | LatinCy +1.2 points with `label_normalization: [v_to_u, j_to_i]` (PROIEL +5.2, Perseus +3.7); Morpheus +3.9 points |
| Lemma case: CIRCSE, ITTB and UDante lowercase all lemmas, including proper names; other treebanks and LatinCy capitalize them | UD Latin | LatinCy another +1.0 points with `lowercase` (CIRCSE +2.8) |
| NFD input: Greek models are trained on NFC | greCy, Greek fastText (Morpheus before the fix) | greCy lemma accuracy 0.92 → 0.28, UPOS identical on only 47% of tokens; fastText coverage 70% → 0% |
| Case of Greek words: the Greek fastText vocabulary is lowercased, AGREE is not | AGREE, Greek fastText | Coverage 70.2% → 82.7% when matched case-insensitively (`case_insensitive: true`) |
| Oxia instead of tonos, sigma variants, Latin letters inside Greek words, stray combining marks | Herodotos (Pliny), NER, UD Greek Perseus lemmas | Few tokens; `nfc`, `final_sigma`, `lunate_sigma` repair the forms. The affected gold lemmas cannot be matched by any model |

No resource is in NFD; all are (almost entirely) NFC. The practical problems are the characters that NFC does
*not* unify: the elision marks, sigma variants, u/v/j and case. NFC even hides elision in one case: UD Perseus
writes the elision mark as U+0313, which NFC merges with a preceding ρ into ῤ (`παῤ` for παρά).

## Method

Every dataset was loaded through SEFLAG with the pinned revisions of the example configurations, so the figures
describe exactly the strings that SEFLAG passes to models and compares with predictions. The dataset statistics
can be reproduced with `seflag inspect-encoding <config> [--by-subset] [--model] [--json]`
(`seflag.encoding`). It reports, for forms, lemmas, lexical items and embedding vocabularies:

* the normalization form;
* characters that NFC changes, and combining marks NFC cannot compose;
* apostrophe-like characters and elided words;
* Greek punctuation characters;
* sigma variants;
* iota subscript and U+1FBE;
* grave accents and unaccented Greek words;
* tokens that mix Greek and Latin (or other) letters;
* v, j, macrons, ligatures and case;
* invisible characters.

Models were examined through their vocabularies, tokenizer configurations and source code, through probes with
escape-coded strings, and by running them on the gold tokens with one variable changed (NFC vs NFD input, one
elision mark replaced by another, input and label normalizations). The accuracies come from SEFLAG runs on the
complete test sets unless a sample is mentioned.

Environment: Python 3.11.9 (Unicode 14.0), spaCy 3.7.6, transformers 4.36.2, gensim 4.3.2, flair 0.13.1,
betacode 1.0, Morpheus 1.0.3.

## Datasets

### UD Latin 2.15 test sets (`daidalos-project/latin_treebanks_ud_test`, 111,394 tokens)

Forms and lemmas are ASCII apart from a few dozen tokens, and NFC wherever they are not. The treebanks follow
different orthographic conventions:

| Treebank | v in forms | v / j in lemmas | lemma case | other |
|---|---|---|---|---|
| CIRCSE | none (u only) | none | lowercase (1 capitalized lemma) | |
| ITTB | none (u only) | none | lowercase; forms lowercase too | |
| LLCT | yes (1,428 v, 102 V) | no lowercase v/j; 34 V, 42 J | proper names capitalized | 33 tokens without lemma |
| Perseus | yes (760) | v (699) | proper names capitalized | U+0027 apostrophe (47) |
| PROIEL | yes (930) | v (931) | proper names capitalized | Roman numerals with U+0305 COMBINING OVERLINE (`D̅C̅C̅X̅X̅V̅`); one Greek word `μάλ’` |
| UDante | yes (950) | none (2 j) | lowercase | Italian vernacular with ì é ò ù à ï; “ ” « » as tokens |

No treebank uses j in forms. None contains macrons or ligatures.

### UD Ancient Greek 2.15 test sets (`daidalos-project/ancient_greek_treebanks_ud_test`, 44,619 tokens)

All forms are NFC (tonos, not oxia). Punctuation uses U+00B7 MIDDLE DOT and U+003B SEMICOLON, the NFC equivalents of
ano teleia and the Greek question mark. PROIEL contains no punctuation tokens at all. Grave accents in running text
are kept in the forms, as they should be. 847 forms (1.9%) are elided.

| Treebank | elision mark in forms | lemma problems |
|---|---|---|
| Perseus | U+0313 COMBINING COMMA ABOVE (650 forms, e.g. `δ̓`); after ρ, NFC has merged it into U+1FE4 ῤ (41 forms: `παῤ` 29×, `σφόδῤ`, `δεῦῤ`) | 42 lemmas start with a stray U+0313 before a capital (`̓Αλέξανδρος`); 18 contain U+0301 on a letter it cannot compose with (`Δημ́ητριος`); 5 contain Latin letters (`κovmh`, `Λύκa`, `δεῖπνοn`); 10 lemmas are themselves elided forms (`ν̓`); 23 lemmas with grave (`δὲ` 6× next to `δέ` 707×) |
| PROIEL | U+2019 RIGHT SINGLE QUOTATION MARK (95) | 50 lemmas with grave (names such as `Λὼτ`, but also `τὶς`, `ἰδοὺ`); 1 lemma with Latin K (`Kίος`) |
| PTNK | U+1FBF GREEK PSILI (67) | 7 lemmas are elided forms (`δεῦρ᾿`, `δώδεκ᾽`); 2 with grave |

Elided words do not always end in a consonant: `δι’` (διά), `δύ̓` (δύο) and `βασιλέ̓` keep a vowel before the
mark.

### Yousef et al. NER test data (`UGARIT/flair_grc_multi_ner`, `test.tsv`, 40,314 tokens)

* **NFC:** forms are NFC except 22 tokens:
  * 11 use U+1FBE GREEK PROSGEGRAMMENI as a standalone iota (`ἐνι`, `ὡσι`). NFC turns it into a plain ι.
  * 6 Greek numerals with U+0374 GREEK NUMERAL SIGN (`αʹ`).
  * 5 forms with U+1FEF GREEK VARIA.
* **Elision marks are mixed within the file:** U+02BC (526 at the end of a word) and U+1FBD GREEK KORONIS (75 at
  the end of a word, 17 inside one, 72 as separate tokens). U+2019 occurs 975 times, always as a separate token,
  i.e. as a quotation mark. There are also U+02BD (18), U+0027 (5) and a backtick (6).
* **Sigma:** 16 words end in σ instead of ς (`δοκεῖσ`, `σῴζουσ`).
* **Other:** fullwidth parentheses, and one leaked placeholder (`᾽ENT_6953`).

### Herodotos Latin NER data (`daidalos-project/Herodotos_dataset`, 95,482 tokens)

* **Latin text:** ASCII with v and V throughout. Greek names in Latin spelling use diaeresis (`aëre`, `Creüsae`,
  `Cybeleïa`).
* **Greek words, encoded differently per file:**
  * PlinyElder: 36 of its 57 Greek tokens use oxia instead of tonos, so they are not NFC (`ἐγκυκλίου` with U+1F77).
  * PlinyYounger: 23 of its 106 Greek tokens use the Latin capital letter C as a lunate sigma (`διόC`,
    `λογισάμενοC`). It also uses U+1FBD as the elision mark, and U+1FFE GREEK DASIA as a spacing breathing
    (`῾ελλάδα`).
* **Corrupted token:** Ovid contains two tokens with annotation that leaked into the form
  (`MenelaÉPOSft-lemma=<unknown>`).

### AGREE (Zenodo 8027490; similarity and retrieval benchmarks, 3,954 Greek words)

Entirely NFC, with iota subscript and no elision. About 9% of the words are capitalized (proper names), which
matters for the lowercased fastText vocabulary (see below).

### Latin synonym selection benchmark (Zenodo 4022818, 13,795 words)

ASCII, lowercase, u/i spelling. Exceptions: `poëma`, `poësis` and `poëta` use ë, and `sсriptor` contains a Cyrillic
с (U+0441) instead of a Latin c, so it can never match a vocabulary entry.

## Models

### Model cards

`documentation/model_cards/` contains cards for four of the resources used here:

* `latincy.md`: `la_core_web_lg`;
* `grecy.md`: `grc_proiel_lg` and `grc_proiel_trf`;
* `flair_grc_bert_ner.md`: `UGARIT/flair_grc_bert_ner`;
* `flair_grc_multi_ner.md`: `UGARIT/flair_grc_multi_ner`, whose repository SEFLAG uses as the source of the NER
  test data.

None of the cards states the encoding a model expects or produces: normalization form, elision mark, u/v and i/j,
case, or the normalization the model applies itself. The cards do list the training data, which explains some of
the behaviour below. greCy was trained on the UD Perseus and PROIEL treebanks (and the TLG), and copies their
elision conventions into its lemmas. LatinCy was trained on UD treebanks with different u/v conventions and
normalizes them away (`normer`). There are no cards yet for `UGARIT/grc-ner-bert`, `UGARIT/grc-ner-xlmr`, the two
fastText models and Morpheus.

### LatinCy `la_core_web_lg` 3.7.4

* **`normer`:** the first pipeline component replaces v→u and j→i (both cases) in `token.norm_`, which the model uses
  as a feature (`la_core_web_lg/functions.py`). Input spelling with u/v or i/j therefore makes no difference to
  tagging and parsing.
* **Lemma output:** always in u/i spelling (`vox` → `uox`, `Vergilius` → `Uergilius`, `iuvenis` → `iuuenis`).
  Proper names are capitalized. Characters the model does not change are copied from the form, including macrons
  (`canō` → `canō`) and NFD sequences.
* **Vectors:** floret (hashed subwords), so there is no vocabulary to inspect.
* **Sensitivity:** NFD input made no difference (Latin is ASCII): lemma accuracy 0.8785 for both on a 1,500-sentence
  sample.
* **Measured on UD Latin, lemma accuracy:**

  | `label_normalization` | total | CIRCSE | ITTB | LLCT | Perseus | PROIEL | UDante |
  |---|---|---|---|---|---|---|---|
  | none (example config) | 0.8905 | 0.7865 | 0.9783 | 0.9654 | 0.7768 | 0.8695 | 0.8233 |
  | `v_to_u`, `j_to_i` | 0.9024 | 0.7865 | 0.9783 | 0.9681 | 0.8138 | 0.9219 | 0.8321 |
  | `v_to_u`, `j_to_i`, `lowercase` | 0.9122 | 0.8141 | 0.9783 | 0.9693 | 0.8347 | 0.9248 | 0.8515 |

### greCy `grc_proiel_lg` 3.7.5

* **Input:** expects NFC. On a 1,500-sentence sample with identical tokens, NFD input lowers lemma accuracy from 0.92
  to 0.28, and only 47% of the UPOS tags stay the same. The tokenizer and the floret vectors see different
  character sequences.
* **Elision:** recognized only with U+02BC MODIFIER LETTER APOSTROPHE. Lemma accuracy on the elided tokens of the UD
  test sets, when their mark is replaced by each variant (forms with ῤ not included):

  | treebank (mark in data) | as is | U+2019 | U+1FBD | U+0313 | U+1FBF | U+0027 | U+02BC |
  |---|---|---|---|---|---|---|---|
  | Perseus (U+0313), 646 tokens | 0.034 | 0.000 | 0.000 | 0.034 | 0.000 | 0.000 | 0.896 |
  | PROIEL (U+2019), 95 tokens | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
  | PTNK (U+1FBF), 67 tokens | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |

  With `input_normalization: [elision_to_u02bc]`, which also turns `παῤ` into `παρʼ`, lemma accuracy on the whole
  UD Greek test set rises from 0.9056 to 0.9226: Perseus 0.8730 → 0.9015, PROIEL 0.9591 → 0.9663, PTNK 0.9030 →
  0.9091.
* **Lemma output copies the form's encoding:** oxia stays oxia (`λόγος` with U+1F79), σ at the end and lunate ϲ
  are kept (`Σωκράτησ`), and U+1FBE is kept. Elided lemmas come out with U+0313, U+1FBF or U+2019, following the
  training data. Some names get a grave accent (`Λὼτ`). A grave in the form is turned into an acute in the lemma
  (`καὶ` → `καί`).

### UGARIT BERT NER models: `UGARIT/grc-ner-bert` (transformers) and `UGARIT/flair_grc_bert_ner` (Flair)

Both use the tokenizer of `pranaydeeps/Ancient-Greek-BERT` (`BertNormalizer`, `lowercase: true`, which also strips
accents). The vocabulary (35,000 pieces) contains no diacritics and no capitals.

* **Insensitive to:** NFC vs NFD, oxia vs tonos, accents, breathings and case. `λόγος`, `λογος` and `Λόγος` all
  become `λογος`.
* **Elision marks:** any word containing U+02BC, U+1FBD, U+1FBF, U+02BD or U+0374 becomes a single `[UNK]`. That
  affects 719 tokens of the NER test data (1.8%). U+2019 becomes a separate piece, and U+0313 is removed silently.
* **Effect on NER:** `input_normalization: [elision_to_u2019]` leaves only 26 `[UNK]` tokens, but entity-level F1
  of `UGARIT/grc-ner-bert` does not improve (0.9857 → 0.9849). The elided words are function words, not entities.
* **Sigma:** σ at the end of a word is tokenized differently from ς (`λογο ##σ`).

### `UGARIT/grc-ner-xlmr`

The SentencePiece tokenizer applies NFKC internally.

* **Equivalent:** NFC, NFD and oxia.
* **Different:** accents and case (`λογος` and `Λόγος` produce other pieces than `λόγος`).
* **Elision marks:** NFKC turns U+1FBD and U+1FBF into a space plus U+0313, which splits the word. U+02BC and U+2019
  are kept as separate pieces. The vocabulary contains 5,174 Greek pieces, 3,194 of them with diacritics.

### Greek fastText CBOW, dim. 50 (Zenodo 7630945)

The vocabulary has 381,105 types.

* **Normalization:** NFC (4 exceptions).
* **Case:** lowercase (35 capitalized types).
* **Elision:** marked with U+0027 (2,768 types).
* **Variants stored as separate types:** grave and acute forms (22,630 types with grave); lunate sigma (8,831 types);
  U+0374/U+0375 numerals (850 types).
* **Coverage of the AGREE words:**
  * 70.2% as they are;
  * 82.7% after lowercasing (`case_insensitive: true`, already mentioned in the example configurations);
  * 0% in NFD.

### LASLA fastText CBOW, dim. 100 (`allLASLA-lemmi-fast-100-CBOW-win10-min5`)

The vocabulary consists of 11,327 lemmas: ASCII, lowercase, u/i spelling (no v, no j). This matches the synonym
benchmark. Its coverage of 87.8% does not change under NFC, NFD, lowercasing or u/v normalization; the missing 12%
are lexical gaps.

### Morpheus 1.0.3, through SEFLAG's `morpheus` backend

* **Greek input conversion:** forms are converted to beta code with the `betacode` package. Morpheus's lemmas are
  converted back and normalized to NFC by SEFLAG, because `betacode` produces oxia.
  * `betacode` handles oxia, σ at the end of a word and lunate ϲ (all become beta code `s`), and grave accents.
  * `betacode` only converts precomposed characters, and only recognizes U+2019 and U+0027 as the elision mark. The
    backend therefore normalizes Greek input to NFC (which also turns U+1FBE into ι) and writes every elision mark
    as `'`, including the ῤ of UD Perseus. Before this fix, NFD input and words with U+0313, U+1FBF, U+1FBD or
    U+02BC were unknown. Lemma accuracy on UD Greek has risen from 0.8613 to 0.8755 (Perseus 0.8682 → 0.8957, PTNK
    0.8444 → 0.8500), and the number of unanalysed words has fallen from 2,058 to 1,313.
  * Capitalized common nouns (`Λόγος` at the start of a sentence) are unknown, because a capital in beta code marks
    a proper name.
* **Latin:**
  * Input: case-insensitive, and accepts u or v and i or j. Macrons make a word unknown.
  * Lemmas are in v/j spelling (`vir`, `juvenis`) with homonym numbers.
  * Lemma accuracy on UD Latin with `strip_homonym_digits` is 0.6912. Adding `v_to_u` and `j_to_i` raises it to
    0.7305: CIRCSE +7.1, UDante +5.2, ITTB +4.2 points.

## Discrepancies in the example configurations

The example configurations keep their original settings, so that they reproduce the published numbers. These
options remove the discrepancies:

| Example | Discrepancy | Remedy |
|---|---|---|
| `latincy_lemma_ud`, `matrix_lemma_ud` | gold u/v, j and lemma case differ per treebank; LatinCy outputs u/i | `label_normalization: [v_to_u, j_to_i]` (+ `lowercase`) |
| `morpheus_lemma_ud_latin` | Morpheus outputs v/j lemmas | add `v_to_u`, `j_to_i` to `label_normalization` |
| `grecy_*_ud` | elision marks of the data are not U+02BC | `input_normalization: [elision_to_u02bc]` |
| `morpheus_lemma_ud_greek` | elision marks, NFD | handled by the backend |
| `ugarit_*_ner_yousef`, `flair_grc_bert_ner_yousef*` | U+02BC/U+1FBD/U+1FBF → `[UNK]` (BERT); word splits (XLM-R) | `input_normalization: [elision_to_u2019]` (BERT; no effect on F1) |
| `grecy_ner_yousef` | mixed elision marks; U+1FBE iota; σ at word end | `input_normalization: [nfc, elision_to_u02bc, final_sigma]` |
| `latincy_ner_herodotos` | Greek words with oxia or Latin C | `input_normalization: [nfc, lunate_sigma]` |
| `agree_*_grc_fasttext` | lowercased vocabulary | `case_insensitive: true` |
| `lasla_synonyms_latin` | none of importance (1 Cyrillic letter) | — |

## Normalizers and tools (implemented 2026-11-18)

* **Elision normalizers**, one per target character, because the models disagree: `elision_to_u02bc` (greCy),
  `elision_to_u2019` (Morpheus, Ancient-Greek-BERT), `elision_to_apostrophe` (Greek fastText). They change the
  mark of elided Greek words only, i.e. a mark at the end of a word directly after a Greek letter (also a vowel:
  `δι’`). Latin apostrophes (`l'`) and marks that stand alone as tokens (quotation marks) stay. U+0313 only counts
  if NFC cannot compose it with the letter before, so a smooth breathing in decomposed text (NFD `οὐ`) stays. A
  final ῤ after another letter becomes ρ + mark.
* **`final_sigma`:** σ at the end of a Greek word → ς.
* **`lunate_sigma`:** ϲ → σ, or ς at the end of a word, and Ϲ → Σ. A Latin C or c inside a word that otherwise only
  has Greek letters is treated the same way (Herodotos `διόC` → `διός`).
* **Morpheus backend:** Greek input is NFC-normalized, and all elision marks become beta code `'` (see above).
* **`seflag inspect-encoding`:** the statistics of this file for any configuration, including matrices. Use
  `--by-subset` for treebanks, `--model` for embedding vocabularies, and `--json` for the full counts with
  examples. SpaCy, transformer and Flair models have no vocabulary to read out; they were examined as described
  under Method.

All normalizers can be used as `input_normalization` (applied to the tokens the model sees) or
`label_normalization` (applied to gold and predicted labels), and are recorded with the results.

## Datasheets and model cards

Integrating this information into the datasheets and model cards is sensible. The datasheets only state "UTF-8",
and the model cards say nothing about encoding, yet the measurements above show differences of 1–5 accuracy points
that depend only on code points, and in one case a drop from 0.92 to 0.28. A fixed set of fields would make these
conventions visible before an evaluation is run:

* **Datasets:** normalization form; Greek: accent forms (tonos/oxia), elision mark, final and lunate sigma, iota
  subscript or adscript, punctuation characters; Latin: u/v and i/j in forms and lemmas, macrons, ligatures; case of
  lemmas; known exceptions (mixed scripts, stray combining marks).
* **Models:** expected input (normalization form, elision mark, case), normalization the model or tokenizer applies
  itself (LatinCy's `normer`, BERT's accent stripping, XLM-R's NFKC), the conventions of its output (lemma
  spelling, case, homonym numbers), and the SEFLAG normalizers that match it (e.g. greCy: `elision_to_u02bc`).

The statistical part can be generated with `seflag inspect-encoding`, and the interpretation written by hand.
Hand-written statements alone would go out of date, and the examples in this file show that the actual characters
often differ from what a dataset's documentation would lead one to expect.

## Not covered

* **spaCy pipelines listed only as installation hints in `backends/spacy.py`:** the other LatinCy sizes, the other
  greCy and OdyCy pipelines, and the transformer variant `grc_proiel_trf`. They are not referenced by any
  configuration, although `grecy.md` describes `grc_proiel_trf`. The other LatinCy sizes probably share the
  `normer` component, but that was not verified.
* **Datasets with datasheets but no SEFLAG configuration**, such as PapyGreek.
