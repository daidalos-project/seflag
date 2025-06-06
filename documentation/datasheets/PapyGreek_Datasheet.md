# Datasheet Papy Greek Treebank
 
### For what purpose was the dataset created? Was there a specific task in mind? Was there a specific gap that needed to be filled? Please provide a description.

- The data is part of the Project: Digital Grammar of Greek Documentary Papyri

### Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?

- Project PI: Marja Vierros
- see: https://www.helsinki.fi/en/researchgroups/digital-grammar-of-greek-documentary-papyri/people
- and for contributors see: https://papygreek.hum.helsinki.fi/people

### Who funded the creation of the dataset? If there is an associated grant, please provide the name of the grantor and the grant name and number.

- ERC starting Grant funded project 2018–2023 (grant agreement No 758481), see: https://papygreek.hum.helsinki.fi/

### Any other comments?

- No

### What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)? Are there multiple types of instances (e.g., movies, users, and ratings; people and interactions between them; nodes and edges)? Please provide a description.

- The documents are annoated texts of post-classical greek papyri ( ca. 300 BCE to 700 CE) see: https://www.helsinki.fi/en/researchgroups/digital-grammar-of-greek-documentary-papyri/about-the-papygreek-project

### How many instances are there in total (of each type, if appropriate)?

- for up-to-date info on corpus size refer to: https://github.com/papygreek/papygreek-treebanks/blob/main/README.md

### Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set? If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set (e.g., geographic coverage)? If so, please describe how

- The corpus is still growing, aiming at containig the whole greek papyrological corpus in the end

### What data does each instance consist of? “Raw” data (e.g., unprocessed text or images) or features? In either case, please provide a description.

- linguistically annotated text in xml format (morphology and syntax)

### Is there a label or target associated with each instance? If so, please provide a description.

- for the Papy Greek annotation guidelines refer to https://github.com/papygreek/papygreek-treebanks
- for the Ancient Greek Dependenc Treebank Guideline refer to https://github.com/PerseusDL/treebank_data/blob/master/AGDT2/guidelines/Greek_guidelines.md#int

### Is any information missing from individual instances? If so, please provide a description, explaining why this information is missing (e.g., because it was unavailable). This does not include intentionally removed information, but might include, e.g., redacted text.

- Due to the fragmentary nature of the greek papyrological record, the instances naturally miss information.
- This is partly accounted for in the annotation following the Leiden Conventions to mark restored or corrected passages.

### Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)? If so, please describe how these relationships are made explicit.

- No

### Are there recommended data splits (e.g., training, development/validation, testing)? If so, please provide a description of these splits, explaining the rationale behind them.

- No

### Are there any errors, sources of noise, or redundancies in the dataset? If so, please provide a description.

- The data have all been manually annotated by experts, but errors of annotation cannot be ruled out
- Since the data consists of natural text it will contain naturally occurring redundancies

### Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)? If it links to or relies on external resources, a) are there guarantees that they will exist, and remain constant, over time; b) are there official archival versions of the complete dataset (i.e., including the external resources as they existed at the time the dataset was created); c) are there any restrictions (e.g., licenses, fees) associated with any of the external resources that might apply to a dataset consumer? Please provide descriptions of all external resources and any restrictions associated with them, as well as
### links or other access points, as appropriate. 

- the dataset is self-contained

### Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor–patient confidentiality, data that includes the content of individuals’ non-public communications)? If so, please provide a description.

- No, the dataset only contains annotations

### Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety? If so, please describe why. If the dataset does not relate to people, you may skip the remaining questions in this section.

- not applicable

### Does the dataset identify any subpopulations (e.g., by age, gender)? If so, please describe how these subpopulations are identified and provide a description of their respective distributions within the dataset.

- A number of ethnic groups of antiquity are referred to.

### Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset? If so, please describe how.

- Only historical individuals

### Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals race or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)? If so, please provide a description.

- Only historical individuals

### Any other comments?

- No

### How was the data associated with each instance acquired? Was the data directly observable (e.g., raw text, movie ratings), reported by subjects (e.g., survey responses), or indirectly inferred/derived from other data (e.g., part-of-speech tags, model-based guesses for age or language)?

- The data consists of publicly available texts and text availabel in scholarly editions

### If the data was reported by subjects or indirectly inferred/derived from other data, was the data validated/verified? If so, please describe how. 

- not applicable

### What mechanisms or procedures were used to collect the data (e.g., hardware apparatuses or sensors, manual human curation, software programs, software APIs)? How were these mechanisms or procedures validated?

- manual human curation
- relying on prievious digital collections publicly available on Papyrological Navigator (see: https://github.com/papyri/idp.data)
- see: Marja Vierros, Erik Henriksson. Preprocessing Greek Papyri for Linguistic Annotation. Journal of Data Mining and Digital Humanities, 2017, Special Issue on Computer-Aided Processing of Intertextuality in Ancient Languages, ⟨10.46298/jdmdh.1385⟩. ⟨hal-01279493v2⟩

### If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?

- Überlieferungszufall 

### Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?

- Project PI: Marja Vierros
- see: https://www.helsinki.fi/en/researchgroups/digital-grammar-of-greek-documentary-papyri/people
- and for contributors see: https://papygreek.hum.helsinki.fi/people

### Over what timeframe was the data collected? Does this timeframe match the creation timeframe of the data associated with the instances (e.g., recent crawl of old news articles)? If not, please describe the timeframe in which the data associated with the instances was created. Were any ethical review processes conducted (e.g., by an institutional review board)? If so, please provide a description of these review processes, including the outcomes, as well as a link or other access point to any supporting documentation. If the dataset does not relate to people, you may skip the remaining questions in this section.

- ongoing project

### Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)? Were the individuals in question notified about the data collection? If so, please describe (or show with screenshots or other information) how notice was provided, and provide a link or other access point to, or otherwise reproduce, the exact language of the notification itself. 

- not applicable

### Did the individuals in question consent to the collection and use of their data? If so, please describe (or show with screenshots or other information) how consent was requested and provided, and provide a link or other access point to, or otherwise reproduce, the exact language to which the individuals consented.

- not applicable

### If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses? If so, please provide a description, as well as a link or other access point to the mechanism (if appropriate). 

- not applicable

### Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted? If so, please provide a description of this analysis, including the outcomes, as well as a link or other access point to any supporting documentation.

- not applicable

### Any other comments?

- No

### Preprocessing/cleaning/labeling

### Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)? If so, please provide a description. If not, you may skip the remainder of the questions in this section.

- Yes. Quote form the readme : "The syntactic trees have been semi-manually annotated by different people (indicated in the metadata tag <annotator>) using the Arethusa editor, and each text has gone through a review process (review board: Marja Vierros and Polina Yordanova). We have followed the Ancient Greek Dependency Treebank Guidelines 2.0 (Celano 2014) for the morphological and syntactic layer (the advanced syntactic/semantic layer is not included). Additional PapyGreek Guidelines are here. All morphological information is in the nine-place string of the <postag> (key). The morphological data from which the annotator could select the correct form was first provided by Morpheus in Arethusa; later we were provided the automatically parsed morphological data by Alek Keersmaekers, for which occasional corrections have been done." (https://github.com/papygreek/papygreek-treebanks/blob/main/README.md)

### Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)? If so, please provide a link or other access point to the “raw” data. 

- raw data are independently and freely available

### Any other comments? 

- No

### Uses

### Has the dataset been used for any tasks already? If so, please provide a description.

- historical linguistics research

### Is there a repository that links to any or all papers or systems that use the dataset? If so, please provide a link or other access point. What (other) tasks could the dataset be used for?

- The data can be used for any text based research (literature, history, anthropology, linguistcs, nlp)
- The data can be used to train nlp tools

### is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses? For example, is there anything that a dataset consumer might need to know to avoid uses that could result in unfair treatment of individuals or groups (e.g., stereotyping, quality of service issues) or other risks or harms (e.g., legal risks, financial harms)? If so, please provide a description. Is there anything a dataset consumer could do to mitigate these risks or harms?

- The data is published under Creative Commons Attribution Share Alike 4.0 International

### are there tasks for which the dataset should not be used? If so, please provide a description.

- Tasks not relying on written language

### any other comments?

- No

### distribution

### will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? If so, please provide a description.
### how will the dataset will be distributed (e.g., tarball on website, API, GitHub)? 

- Creative Commons Attribution Share Alike 4.0 International
- data is avaialble from: https://github.com/papygreek and https://papygreek.hum.helsinki.fi/annotated/documentary

### does the dataset have a digital object identifier (DOI)?

- https://zenodo.org/records/8428823

### when will the dataset be distributed?

- it already is available

### will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)? If so, please describe this license and/or ToU, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms or ToU, as well as any fees associated with these restrictions.

- Creative Commons Attribution Share Alike 4.0 International

### have any third parties imposed IP-based or other restrictions on the data associated with the instances? If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms, as well as any fees associated with these restrictions.

- 

### do any export controls or other regulatory restrictions apply to the dataset or to individual instances? If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any supporting documentation.

- 

### any other comments?

- NO

### maintenance

### who will be supporting/hosting/maintaining the dataset?

- Department of Languages at the University of Helsinki
- see: https://zenodo.org/records/8428823 and https://www.helsinki.fi/en/researchgroups/digital-grammar-of-greek-documentary-papyri/about-the-papygreek-project

### how can the owner/curator/manager of the dataset be contacted (e.g., email address)?

- https://www.helsinki.fi/en/researchgroups/digital-grammar-of-greek-documentary-papyri/people

### is there an erratum? If so, please provide a link or other access point. Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)? If so, please describe how often, by whom, and how updates will be communicated to dataset consumers (e.g., mailing list, GitHub)?

- dataset is updated constantly
- see: https://papygreek.hum.helsinki.fi/

### if the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were the individuals in question told that their data would be retained for a fixed period of time and then deleted)? If so, please describe these limits and explain how they will be enforced.

- not applicable

### will older versions of the dataset continue to be supported/hosted/maintained? If so, please describe how. If not, please describe how its obsolescence will be communicated to dataset consumers.

- see: https://papygreek.hum.helsinki.fi/

### if others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so? If so, please provide a description. Will these contributions be validated/verified? If

- see: https://papygreek.hum.helsinki.fi/help

### any other comments?

- No