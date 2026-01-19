NOTE: This template for datasheets for ancient language data is based on the proposal by Gebru et al. 2021 https://arxiv.org/abs/1803.09010.
The majority of questions is taken from there, but in a slightly rearanged order. However, as some questions are not relevant for historical data, they have been left out. Questions which are of relevance for Humanities scholars researching ancient languages have been added.
The question have been answered to the best of the knowledge of the Daidalos Team.

# Abbreviations

AGDT = Ancient Greek Dependency Treebank = Perseus Treenbank
Perseus = Perseus Treebank
PTNK = UD Ancient Greek PTNK
PROIEL = Pragmatic Resources of Old Indo-European Languages, UD Ancient Greek PROIEL
SEFLAG = Systematic Evaluation Framework for natural language processing models and datasets in Latin and Ancient Greek
UD = Universal Dependencies

# Dataset: grc_ud_test_dataset_for_use_of_SEFLAG

**All links in this document have last been checked in Sept. 2025.**

# Part I: Dataset Creation and Content 

## Dataset Creation and Content (technical)
 
### 1. For what purpose was the dataset created? Was there a specific task in mind? Was there a specific gap that needed to be filled? Please provide a description.

- The dataset consists the test splits of all greek treebanks part of version 2.15 of the Universal Dependencies Project (see below), namely: the PTNK, PROIEL and Perseus (AGDT) Treebanks. The dataset was created to serve as a goldstandard for the evaluation of classical language NLP tools as part of SEFLAG 
- The original goals of the UD treebanks (abridged from https://universaldependencies.org/introduction.html#introduction (as of Sept. 2025)): The Universal Dependencies (UD) treebanks were created to facilitate multilingual syntactic analysis and parsing by providing a standardized framework for dependency structures across languages. The goal was to support a broad range of tasks such as natural language understanding, machine translation, and cross-lingual studies, while addressing the lack of consistent linguistic annotations across diverse languages. 
  "The annotation scheme is based on an evolution of (universal) Stanford dependencies (de Marneffe et al., 2006, 2008, 2014), Google universal part-of-speech tags (Petrov et al., 2012), and the Interset interlingua for morphosyntactic tagsets (Zeman, 2008)." https://universaldependencies.org/introduction.html#introduction (as of Sept. 2025)  

### 2. Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?

- grc_ud_test_dataset_for_use_of_SEFLAG was compiled out of existing treebanks by the Daidalos team
  - see: https://daidalos-projekt.de/
- the UD data was created by different researchgroup for each treebank (see below).
  - UD main coordinator: Joakim Nivre, see https://universaldependencies.org/introduction.html#project-organization (Sept. 2025)
- the Perseus (AGDT) data was created by Tufts and Leipzig Universities, main contributers: Giuseppe G. A. Celano, Gregory Crane, Bridget Almas & al., see http://perseusdl.github.io/treebank_data/ (Sept. 2025)
  - UD conversion by Giuseppe G. A. Celano, see https://universaldependencies.org/treebanks/grc_perseus/index.html#acknowledgments (contributers also listed there) (Sept. 2025)
- PROIEL:
  - citation: Dag T. T. Haug and Marius L. Jøhndal. 2008. 'Creating a Parallel Treebank of the Old Indo-European Bible Translations'. In Caroline Sporleder and Kiril Ribarov (eds.). Proceedings of the Second Workshop on Language Technology for Cultural Heritage Data (LaTeCH 2008) (2008), pp. 27-34.
  - official website: https://dev.syntacticus.org (Sept. 2025)
  - UD conversion by Dag Haug, see: https://universaldependencies.org/treebanks/grc_proiel/index.html#acknowledgments (Sept. 2025)
- PTNK
  - citation: Daniel G. Swanson, Bryce D. Bussert, and Francis Tyers. 2024. Producing a Parallel Universal Dependencies Treebank of Ancient Hebrew and Ancient Greek via Cross-Lingual Projection. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 13074–13078, Torino, Italia. ELRA and ICCL.

### 3. Who funded the creation of the dataset? If there is an associated grant, please provide the name of the grantor and the grant name and number.

**Reicht die Angabe für Daidalos oder besser für alles recherchieren?**

- Daidalos project funded by DFG
- "UD is an open collaboration with many project members." https://universaldependencies.org/introduction.html#project-organization, refer to the linked site for further information
- AGDT funded
- PROIEL funded by
- PTNK funded by

### 4. What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)? Are there multiple types of instances (e.g., movies, users, and ratings; people and interactions between them; nodes and edges)? Please provide a description.

- The dataset consists of syntactically annotated sentences from a wide variety of languages. Each instance represents a sentence broken down into individual tokens (words or morphemes) with syntactic dependencies between them.

### 5. How many instances are there in total (of each type, if appropriate)?

- Sentences:
- Tokens:

### 6. Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set? If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set (e.g., geographic coverage)? If so, please describe how

- The data used in the grc_ud_test_dataset_for_use_of_SEFLAG comprises the test splits of the Universal Dependencies Treebanks for Ancient Greek and thus represent a sample of the full data available at UD. Train and dev splits are also available at the UD website for each treebank (refer to the treebank hub pages linked above).
- The contents of the UD treebanks are themselves samples from the corpus of the surviving ancient greek language material. See also section IV for more details on the content.

### 7. What data does each instance consist of? “Raw” data (e.g., unprocessed text or images) or features? In either case, please provide a description.

- Treebanks are stored as plain text files encoded with UTF-8 in the .conllu format (see: https://universaldependencies.org/format.html)
- Each file contains a number of sentences in tokenized form with linguistic annotations

### 8. Is there a label or target associated with each instance? If so, please provide a description.

- There are multiple labels associated with each instance conveying a range of linguistic as well as meta information for each sentence and each token inside a sentence. Refer to the UD guidelines and the treebank specific sites linked above.

### 9. Is any information missing from individual instances? If so, please provide a description, explaining why this information is missing (e.g., because it was unavailable). This does not include intentionally removed information, but might include, e.g., redacted text.

- No information is missing from the instances, but not all annotations possible in the UD annotation scheme have been made (e.g. the treebanks usually do not contain Enhanced Dependencies). Additionally, not all fields existing in the UD annotation format have been filled for every token because not all tokens require every field.


### 10. Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)? If so, please describe how these relationships are made explicit.

- Each sentence is assigned an ID  which marks its position in the whole corpus
- Each token inside a sentence is assigned a syntactic dependency relationship with it's head word.

### 11. Are there recommended data splits (e.g., training, development/validation, testing)? If so, please provide a description of these splits, explaining the rationale behind them.

**Haben wir eigentlich nicht, oder?**
- Test, dev and train splits can be created via hugging face datasets as for any other dataset.
- However, keep in mind, that the dataset was compiled with the goal to serve as a gold standard for testing NLP tools

### 12. Are there any errors, sources of noise, or redundancies in the dataset? If so, please provide a description.

- Since the data consists of natural text it will contain naturally occurring redundancies

### 13. Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)? 

- the dataset is self-contained

#### If it links to or relies on external resources:
#### 13 a) are there guarantees that they will exist, and remain constant, over time; 
#### 13 b) are there official archival versions of the complete dataset (i.e., including the external resources as they existed at the time the dataset was created); 
#### 13 c) are there any restrictions (e.g., licenses, fees) associated with any of the external resources that might apply to a dataset consumer? Please provide descriptions of all external resources and any restrictions associated with them, as well as links or other access points, as appropriate. 

### 14. What mechanisms or procedures were used to collect the data (e.g., hardware apparatuses or sensors, manual human curation, software programs, software APIs)? How were these mechanisms or procedures validated?

**Reicht das so?**

- manual human curation

### 15. If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?

- The dataset was compiled with the goal to serve as a gold standard for testing NLP tools.

### 16. Over what timeframe was the data collected? Does this timeframe match the creation timeframe of the data associated with the instances (e.g., recent crawl of old news articles)?

**Kein match, weil die Daten antik sind? Geht es darum, wann die Annotationen gemacht wurden oder wann die Texte geschrieben wurden?**

- 

#### 16 a) If not, please describe the timeframe in which the data associated with the instances was created. 

### 17. How was the data associated with each instance acquired? Was the data directly observable (e.g., raw text, movie ratings), reported by subjects (e.g., survey responses), or indirectly inferred/derived from other data (e.g., part-of-speech tags, model-based guesses for age or language)?

- The data consists of publicly available texts and text available in scholarly editions

### Comments

- None

## Dataset Creation and Content (legal, ethical)

### 18. Were any ethical review processes conducted (e.g., by an institutional review board)? If so, please provide a description of these review processes, including the outcomes, as well as a link or other access point to any supporting documentation.

- SEFLAG dataset: no
- Original Treebanks: unknown to the authors of this datasheet

### 19. Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?

- SEFLAG dataset: research assistant and student assistant paid by project fund
- regarding contributers to AGDT see: https://perseusdl.github.io/treebank_data/, https://universaldependencies.org/treebanks/grc_perseus/index.html#acknowledgments (09.2025). The annotators of every sentence are indicated in the .xml files of the original AGDT project. This information is not part of the UD annotations.
- regarding contributers to PROIEL see: https://dev.syntacticus.org/ (09.2025)
- regarding contributers to PTNK refer to: Daniel G. Swanson, Bryce D. Bussert, and Francis Tyers. 2024. Producing a Parallel Universal Dependencies Treebank of Ancient Hebrew and Ancient Greek via Cross-Lingual Projection. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 13074–13078, Torino, Italia. ELRA and ICCL.

### 20. Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor–patient confidentiality, data that includes the content of individuals’ non-public communications)? If so, please provide a description.

**Irrelevant?**

- No, the dataset only contains annotations

### 21. Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety? If so, please describe why.
**Irrelevant?**

- not applicable

### 22. Does the dataset identify any subpopulations (e.g., by age, gender)? If so, please describe how these subpopulations are identified and provide a description of their respective distributions within the dataset.

**Irrelevant?**

- A number of ethnic groups of antiquity are referred to.

### 23. Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset? If so, please describe how.

**Irrelevant?**

- Only historical individuals

### 24. Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals race or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)? If so, please provide a description.

**Irrelevant?**

- Only historical individuals

### 25. If the data was reported by subjects or indirectly inferred/derived from other data, was the data validated/verified? If so, please describe how. 

**Irrelevant?**

- not applicable

### 26. Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)? Were the individuals in question notified about the data collection? If so, please describe (or show with screenshots or other information) how notice was provided, and provide a link or other access point to, or otherwise reproduce, the exact language of the notification itself. 

**Irrelevant?**

- not applicable

### 27. Did the individuals in question consent to the collection and use of their data? If so, please describe (or show with screenshots or other information) how consent was requested and provided, and provide a link or other access point to, or otherwise reproduce, the exact language to which the individuals consented.

**Irrelevant?**

- not applicable

### 28. If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses? If so, please provide a description, as well as a link or other access point to the mechanism (if appropriate). 

**Irrelevant?**

- not applicable

### 29. Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted? If so, please provide a description of this analysis, including the outcomes, as well as a link or other access point to any supporting documentation.

**Irrelevant?**

- not applicable

### Comments

- None

# Part II: Technical

## Preprocessing/cleaning/labeling

### 30. Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)? If so, please provide a description.


- A range of preprocessing steps (such as normalisation, tokenization, lemmatisation) has been conducted in the process of the original annotation for each UD Treebank contained in this dataset. All instances are labeled/linguistically annotated.
- For the grc_ud_test_dataset_for_use_of_SEFLAG no further preprocessing has been done.

### 31. Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)? If so, please provide a link or other access point to the “raw” data. 

- Raw data are independently and freely available
- As part of the UD annotation scheme ech sentence in the .conllu files contains a plain text string containing the full text of the sentence (based on the scholarly editions the sentence was taken from).

### Comments 

- None

## Distribution

### 32. will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? If so, please provide a description.
### how will the dataset will be distributed (e.g., tarball on website, API, GitHub)? 

- The grc_ud_test_dataset_for_use_of_SEFLAG is available on Huggingface at https://huggingface.co/daidalos-project
- The original UD data is avaialble from the universal dependencies website: https://universaldependencies.org/#language-

### 33. Does the dataset have a digital object identifier (DOI)?

**Einfügen, wenn grc dataset online ist**
-

### 34. When will the dataset be distributed?

- The dataset is already distributed.

### 35. Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)? If so, please describe this license and/or ToU, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms or ToU, as well as any fees associated with these restrictions.

**Prüfen**

- grc_ud_test_dataset_for_use_of_SEFLAG: Creative Commons Zero v1.0
- UD Perseus Treebank: CC BY-NC-SA 2.5 (https://universaldependencies.org/treebanks/grc_perseus/index.html#acknowledgments)
- UD PROIEL Treebank: CC BY-NC-SA 3.0 (https://universaldependencies.org/treebanks/grc_proiel/index.html#acknowledgments)
- UD PTNK Treebank: CC BY-SA 4.0 (https://universaldependencies.org/treebanks/grc_ptnk/index.html)

### 36. Have any third parties imposed IP-based or other restrictions on the data associated with the instances? If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms, as well as any fees associated with these restrictions.

**Prüfen**

- Unknown to the authors of this datasheet

### 37. Do any export controls or other regulatory restrictions apply to the dataset or to individual instances? If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any supporting documentation.

**Prüfen**

- No ?
- Unknown to the authors of this datasheet

### 38. Comments?

- No

## Maintenance

### 39. Who will be supporting/hosting/maintaining the dataset?

- The grc_ud_test_dataset_for_use_of_SEFLAG is maintained by the Daidalos Team: https://daidalos-projekt.de/contact/
- The UD Treebanks are maintained by the UD team: https://universaldependencies.org/introduction.html#project-organization 

### 40. How can the owner/curator/manager of the dataset be contacted (e.g., email address)?

- The grc_ud_test_dataset_for_use_of_SEFLAG is maintained by the Daidalos Team: https://daidalos-projekt.de/contact/
- The UD Treebanks are maintained by the UD team: https://universaldependencies.org/introduction.html#project-organization 

### 41. Is there an erratum? If so, please provide a link or other access point. Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)? If so, please describe how often, by whom, and how updates will be communicated to dataset consumers (e.g., mailing list, GitHub)?

- The grc_ud_test_dataset_for_use_of_SEFLAG might be expanded to include treebanks other than UD in the future
- As UD is an ongoing project, the original UD Treebanks can potentially be modified, expanded or updated

### 42. If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were the individuals in question told that their data would be retained for a fixed period of time and then deleted)? If so, please describe these limits and explain how they will be enforced.

**Irrelevant?**

- not applicable

### 43. Will older versions of the dataset continue to be supported/hosted/maintained? If so, please describe how. If not, please describe how its obsolescence will be communicated to dataset consumers.

- for this SEFLAG dataset: No
- the distribution of the original UD data is maintained by the UD team (see above)

### 44. If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so? If so, please provide a description. Will these contributions be validated/verified? If

- for the SEFLAG dataset: No
- For possible contribution to the original UD data refer to the UD website: https://universaldependencies.org/contributing/index.html

### Comments

- No

# Part III: Uses

### 45. Has the dataset been used for any tasks already? If so, please provide a description.

- historical linguistics research
- The data can be used for any text based research (literature, history, anthropology, linguistcs, nlp)
- The data can be used to train nlp tools
- The dataset here provided has been used in the evaluation of NLP tools as part of SEFLAG

### 46. Is there a repository that links to any or all papers or systems that use the dataset? If so, please provide a link or other access point. What (other) tasks could the dataset be used for?

- https://daidalos-projekt.de/resources/

### 47. Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses? For example, is there anything that a dataset consumer might need to know to avoid uses that could result in unfair treatment of individuals or groups (e.g., stereotyping, quality of service issues) or other risks or harms (e.g., legal risks, financial harms)? If so, please provide a description. Is there anything a dataset consumer could do to mitigate these risks or harms?

**Bezieht sich wohl eher auf ethische Fragen und daher nicht so relevant?**

- 

### 48. Are there tasks for which the dataset should not be used? If so, please provide a description.

- Tasks not relying on written language

### Comments

- No

# Part IV: Content (non-technical)

### 49. Which language? Which period in the language history?

- Language: Ancient Greek (grc)
- AGDT: Classical Ancient Greek
- PROIEL: Classical (Herodotos), Koiné (Bible)
- PTNK: Koiné

### 50. Which Region?

- AGDT: Greece
- PROIEL: Greece, Egypt (Alexandria)
- PTNK: Egypt (Alexandria)

### 51. Original Medium (Inscription, Book, ...)?

- books

### 52. Which genre? Which topics? Which authors?

- The content of each UD treebank is listed on the respective treebank hub page. Information is taken from there
- AGDT lists the follwing (source: https://universaldependencies.org/treebanks/grc_perseus/index.html#description 26.09.2025):
  "Aesop 	Fabulae
  Aeschylus 	Agamemnon
  Aeschylus 	Eumenides
  Aeschylus 	Libation Bearers
  Aeschylus 	Prometheus Bound
  Aeschylus 	Persians
  Aeschylus 	Seven Against Thebes
  Aeschylus 	Supplian Women
  Anonymous 	Hymn to Demeter
  Apollodorus 	Library
  Athenaeus 	The Deipnosophists
  Diodorus Siculus 	Bibliotheca Historica
  Herodotus 	Histories
  Hesiod 	Shield of Heracles
  Hesiod 	Theogony
  Hesiod 	Works and Days
  Homer 	Iliad
  Lysias 	Against Pancleon
  Lysias 	Alcybiades 1
  Lysias 	Alcybiades 2
  Lysias 	On the Murder of Eratosthenes
  Plutarch 	Alcibiades
  Plutarch 	Lycurgus
  Plybius 	Histories
  Pseudo-Homer 	Hymn to Demeter
  Sophocles 	Ajax
  Sophocles 	Antigone
  Sophocles 	Electra
  Sophocles 	Oedipus Tyrannus
  Sophocles 	Trachinae
  Thucydides 	Histories"
- PROIEL gives the following information (source https://universaldependencies.org/treebanks/grc_proiel/index.html#description 25.09.2029): "UD_Ancient_Greek-PROIEL is converted from the Ancient Greek data in the PROIEL treebank, and consists of the New Testament plus selections from Herodotus. The Ancient Greek PROIEL treebank is based on the Ancient Greek data from the PROIEL treebank, which is maintained at the Department of Philosophy, Classics, History of Arts and Ideas at the University of Oslo. The conversion is based on the 20180408 release of the PROIEL treebank available from https://github.com/proiel/proiel-treebank/releases. The original annotators are acknowledged in the files available there. The conversion code is available in the Rubygem proiel-cli, https://github.com/proiel/proiel-cli. The treebank contains most of the New Testament plus selections from Herodotus’ Histories. The original annotation guidelines are available at http://folk.uio.no/daghaug/syntactic_guidelines.pdf."
- PTNK gives the following information (source 29.09.2025): "UD Ancient Greek PTNK contains portions of the Septuagint according to the Codex Alexandrinus. This treebank was produced using text extracted from https://greekdoc.github.io with initial syntactic relations produced by word-aligning and projecting the relations from the parallel Ancient Hebrew treebank."

### 53. Based on which editions?

**Steht evtl. irgendwo in den Artikeln.**

- 

### Comments

- No

# Bibliography

  Konstantin Schulz and Florian Deichsler. 2024. SEFLAG: Systematic Evaluation Framework for NLP Models and Datasets in Latin and Ancient Greek. In Proceedings of the 4th International Conference on Natural Language Processing for Digital Humanities, pages 247–258, Miami, USA. Association for Computational Linguistics.

  Bamman, David und Crane, Gregory R. 2011. ‚‚The Ancient Greek and Latin Dependency Treebanks‘‘. In: Sporleder, Caroline und van den Bosch, Aantal u. Zervanou, Kalliopi (Hrsg.). Language Technology for Cultural Heritage. Berlin/Heidelberg. Theory and Applications of Natural Language Processing.

  Dag T. T. Haug and Marius L. Jøhndal. 2008. 'Creating a Parallel Treebank of the Old Indo-European Bible Translations'. In Caroline Sporleder and Kiril Ribarov (eds.). Proceedings of the Second Workshop on Language Technology for Cultural Heritage Data (LaTeCH 2008) (2008), pp. 27-34. 
  
  Daniel G. Swanson, Bryce D. Bussert, and Francis Tyers. 2024. Producing a Parallel Universal Dependencies Treebank of Ancient Hebrew and Ancient Greek via Cross-Lingual Projection. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 13074–13078, Torino, Italia. ELRA and ICCL. 

  Joakim Nivre, Marie-Catherine de Marneffe, Filip Ginter, Jan Hajič, Christopher Manning, Sampo Pyysalo, Sebastian Schuster, Francis Tyers, and Daniel Zeman (2020). Universal Dependencies v2: An Evergrowing Multilingual Treebank Collection. Proceedings of the 12th International Conference on Language Resources and Evaluation (LREC 2020), pp. 4034–4043, Marseille, France.
