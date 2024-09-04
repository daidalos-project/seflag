# Datasheet: Universal Dependencies Treebanks (esp. Latin and Ancient Greek)

## MOTIVATION
For what purpose was the dataset created? Was there a specific task in mind? Was there a specific gap that needed to be filled? Please provide a description.

- The Universal Dependencies (UD) treebanks were created to facilitate multilingual syntactic analysis and parsing by providing a standardized framework for dependency structures across languages. The goal was to support a broad range of tasks such as natural language understanding, machine translation, and cross-lingual studies, while addressing the lack of consistent linguistic annotations across diverse languages.

Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?

- UD is an open community effort with over 600 contributors producing over 200 treebanks in over 150 languages. 
- The annotation scheme is based on an evolution of (universal) Stanford dependencies (de Marneffe et al., 2006, 2008, 2014), Google universal part-of-speech tags (Petrov et al., 2012), and the Interset interlingua for morphosyntactic tagsets (Zeman, 2008). 
- The project is coordinated by Joakim Nivre (aka chief cat herder). 
- Releases (including validation and documentation) are managed by Filip Ginter, Sampo Pyysalo and Dan Zeman. 
- Universal guidelines are managed by a small group of core members, currently consisting of Marie de Marneffe, Filip Ginter, Yoav Goldberg, Jan Hajič, Chris Manning, Ryan McDonald, Lori Levin, Joakim Nivre, Slav Petrov, Sampo Pyysalo, Nathan Schneider, Sebastian Schuster, Natalia Silveira, Reut Tsarfaty, Fran Tyers, Amir Zeldes and Dan Zeman. 
- Language-specific guidelines and treebanks are maintained by each specific language team. 
- Issues are raised on GitHub and resolved through discussion and voting.


Who funded the creation of the dataset? If there is an associated grant, please provide the name of the grantor and the grant name and number.

- unknown

Any other comments?

- No

## COMPOSITION
What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)? Are there multiple types of instances (e.g., movies, users, and ratings; people and interactions between them; nodes and edges)? Please provide a description.

- The dataset consists of syntactically annotated sentences from a wide variety of languages. Each instance represents a sentence broken down into individual tokens (words or morphemes) with syntactic dependencies between them.

How many instances are there in total (of each type, if appropriate)?

- ca. 14 million words in version 2.5 (Nivre, J., de Marneffe, M. C., Ginter, F., Hajic, J., Manning, C. D., Pyysalo, S., ... & Zeman, D. (2020, May). Universal Dependencies v2: An Evergrowing Multilingual Treebank Collection. In Proceedings of the Twelfth Language Resources and Evaluation Conference (pp. 4034-4043).)

Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set? If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set (e.g., geographic coverage)? If so, please describe how this representativeness was validated/verified. If it is not representative of the larger set, please describe why not (e.g., to cover a more diverse range of instances, because instances were withheld or unavailable).

- "The languages in UD v2.5 represent 20 different language families (or equivalent) [...]. The selection is very heavily biased towards Indo-European languages (48 out of 90), and towards a few branches of this family – Germanic (10), Romance (8) and Slavic (13) – but it is worth noting that the bias is (slowly) becoming less extreme over time." (Nivre et al. 2020)

What data does each instance consist of? "Raw" data (e.g., unprocessed text or images) or features? In either case, please provide a description.

- Each treebank includes raw text (i.e., sentences) and annotations for tokenization, segmentation, lemmatization, part-of-speech tags, morphological features, and syntactic dependencies, all following the Universal Dependencies framework.

Is there a label or target associated with each instance? If so, please provide a description.

- Yes, each token in a sentence is labeled with its syntactic head and dependency relation, lemma, part-of-speech tag, and morphological features. See https://universaldependencies.org/guidelines.html .

Is any information missing from individual instances? If so, please provide a description, explaining why this information is missing (e.g., because it was unavailable). This does not include intentionally removed information, but might include, e.g., redacted text.

- No significant information is missing from individual instances, though some languages or treebanks may have fewer annotations for specific linguistic features, depending on the data's availability and quality.

Are relationships between individual instances made explicit (e.g., users' movie ratings, social network links)? If so, please describe how these relationships are made explicit.

- Yes, relationships between tokens (words) within a sentence are made explicit through unique identifiers, e.g., in their dependency relations, linking words to their syntactic heads.

Are there recommended data splits (e.g., training, development/validation, testing)? If so, please provide a description of these splits, explaining the rationale behind them.

- Yes, the dataset provides splits for training, development/validation, and testing. These splits ensure that the model training is done on one portion of the data, while evaluation occurs on unseen data.

Are there any errors, sources of noise, or redundancies in the dataset? If so, please provide a description.

- Some treebanks may contain minor annotation errors or inconsistencies, which are periodically reviewed and corrected by contributors. Additionally, certain languages might exhibit variability in annotations due to the evolving nature of linguistic research.

Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)? If it links to or relies on external resources, a) are there guarantees that they will exist, and remain constant, over time; b) are there official archival versions of the complete dataset (i.e., including the external resources as they existed at the time the dataset was created); c) are there any restrictions (e.g., licenses, fees) associated with any of the external resources that might apply to a dataset consumer? Please provide descriptions of all external resources and any restrictions associated with them, as well as links or other access points, as appropriate.

- The dataset is largely self-contained, although some treebanks may link to external corpora from which they were derived. These links are not guaranteed to persist indefinitely, but they are usually not essential for using the dataset.

Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor--patient confidentiality, data that includes the content of individuals' non-public communications)? If so, please provide a description.

- No, the dataset does not contain confidential data. All texts are sourced from publicly available material such as published literature, documentary texts, etc.

Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety? If so, please describe why. If the dataset does not relate to people, you may skip the remaining questions in this section.

- Some texts may contain culturally sensitive material, offensive language, or politically charged content, as they are derived from authentic sources like literary texts.

Does the dataset identify any subpopulations (e.g., by age, gender)? If so, please describe how these subpopulations are identified and provide a description of their respective distributions within the dataset.

- The dataset includes sentences that describe groups of ancient people by referring to their gender, race, etc.

Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset? If so, please describe how.

- Only historical individuals

Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals race or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)? If so, please provide a description.

- Yes, but only for historical individuals

Any other comments?

- No

## COLLECTION
How was the data associated with each instance acquired? Was the data directly observable (e.g., raw text, movie ratings), reported by subjects (e.g., survey responses), or indirectly inferred/derived from other data (e.g., part-of-speech tags, model-based guesses for age or language)?

- Data for each language treebank was acquired from publicly available sources such as literary text editions. Sentences were manually or semi-automatically annotated by linguists following the Universal Dependencies framework.

If the data was reported by subjects or indirectly inferred/derived from other data, was the data validated/verified? If so, please describe how.

- Annotations have usually been verified by multiple expert annotators, but, to a certain degree, each treebank follows its own guidelines.

What mechanisms or procedures were used to collect the data (e.g., hardware apparatuses or sensors, manual human curation, software programs, software APIs)? How were these mechanisms or procedures validated?

- manual human curation, validated through multi-way annotation (intersubjective agreement)

If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?

- The dataset is sampled based on the availability of high-quality text data for each language. Sampling strategies vary depending on the language and source material but aim to provide broad coverage of linguistic phenomena.

Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?

- The data collection process involved academic researchers, graduate students, and linguists, often working as part of research teams. They were typically compensated through academic salaries or research grants.

Over what timeframe was the data collected? Does this timeframe match the creation timeframe of the data associated with the instances (e.g., recent crawl of old news articles)? If not, please describe the timeframe in which the data associated with the instances was created. 

- Data collection has been ongoing since the inception of the Universal Dependencies project in 2014, with treebanks continuously added and updated.

Were any ethical review processes conducted (e.g., by an institutional review board)? If so, please provide a description of these review processes, including the outcomes, as well as a link or other access point to any supporting documentation. If the dataset does not relate to people, you may skip the remaining questions in this section.

- No formal ethical review processes were typically required, as the dataset is derived from publicly available texts, without collecting sensitive personal information.

Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)? Were the individuals in question notified about the data collection? If so, please describe (or show with screenshots or other information) how notice was provided, and provide a link or other access point to, or otherwise reproduce, the exact language of the notification itself.

- not applicable

Did the individuals in question consent to the collection and use of their data? If so, please describe (or show with screenshots or other information) how consent was requested and provided, and provide a link or other access point to, or otherwise reproduce, the exact language to which the individuals consented.

- not applicable

If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses? If so, please provide a description, as well as a link or other access point to the mechanism (if appropriate).

- not applicable

Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted? If so, please provide a description of this analysis, including the outcomes, as well as a link or other access point to any supporting documentation.

- not applicable

Any other comments?

- No

## Preprocessing/cleaning/labeling

Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)? If so, please provide a description. If not, you may skip the remainder of the questions in this section.

- Yes, the text was segmented, tokenized and labeled with syntactic annotations according to the UD guidelines. This process involved POS tagging, dependency parsing, and labeling morphological features.

Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)? If so, please provide a link or other access point to the "raw" data.

- Yes, the raw texts are typically preserved alongside the preprocessed annotations to ensure future reusability.

Any other comments?

-No

## Uses

Has the dataset been used for any tasks already? If so, please provide a description.

- Yes, the UD treebanks have been widely used for tasks such as syntactic parsing, cross-lingual syntactic analysis, and multilingual language modeling.

Is there a repository that links to any or all papers or systems that use the dataset? If so, please provide a link or other access point.

- Yes, many papers and projects using UD treebanks are referenced on the Universal Dependencies website: https://universaldependencies.org/related.html

What (other) tasks could the dataset be used for?

- The dataset could be used for machine translation, syntactic transfer learning, linguistic typology studies, and cross-lingual sentiment analysis.

Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses? For example, is there anything that a dataset consumer might need to know to avoid uses that could result in unfair treatment of individuals or groups (e.g., stereotyping, quality of service issues) or other risks or harms (e.g., legal risks, financial harms)? If so, please provide a description. Is there anything a dataset consumer could do to mitigate these risks or harms?

- Due to varying quality and completeness of annotations across languages, users should be cautious when using specific treebanks for tasks requiring high precision. Also, most treebanks contain only small amounts of text, which is why they may not be suitable for knowledge transfer scenarios, such as applications to other text genres, language periods, etc.

Are there tasks for which the dataset should not be used? If so, please provide a description.

- No

Any other comments?

- No

## Distribution

Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? If so, please provide a description.

- Yes, the dataset is publicly available for academic and research purposes through the Universal Dependencies website.

How will the dataset be distributed (e.g., tarball on website, API, GitHub)?

- The dataset is distributed as downloadable files via the UD website (https://universaldependencies.org/#download) and treebank-specific GitHub repositories.

Does the dataset have a digital object identifier (DOI)?

- http://hdl.handle.net/11234/1-5502

When will the dataset be distributed?

- The dataset has been available since 2015 and is continuously updated.

Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)? If so, please describe this license and/or ToU, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms or ToU, as well as any fees associated with these restrictions.

- Most treebanks use Creative Commons licenses: https://lindat.mff.cuni.cz/repository/xmlui/page/license-ud-2.14

Have any third parties imposed IP-based or other restrictions on the data associated with the instances? If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms, as well as any fees associated with these restrictions.

- unknown

Do any export controls or other regulatory restrictions apply to the dataset or to individual instances? If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any supporting documentation.

- unknown

Any other comments?

- No

## Maintenance

Who will be supporting/hosting/maintaining the dataset?

- The Universal Dependencies organization is responsible for maintaining and updating the dataset. Individual treebanks are updated by their respective maintainers.

How can the owner/curator/manager of the dataset be contacted (e.g., email address)?

- Contact data is available on Joakim Nivre's personal website: https://jnivre.github.io/

Is there an erratum? If so, please provide a link or other access point. Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)? If so, please describe how often, by whom, and how updates will be communicated to dataset consumers (e.g., mailing list, GitHub)?

- Yes, updates are made periodically as new languages are added and existing treebanks are revised.

If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were the individuals in question told that their data would be retained for a fixed period of time and then deleted)? If so, please describe these limits and explain how they will be enforced.

- not applicable

Will older versions of the dataset continue to be supported/hosted/maintained? If so, please describe how. If not, please describe how its obsolescence will be communicated to dataset consumers.

- Yes, older versions remain accessible through the website and associated repository archives, though updates are encouraged.

If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so? If so, please provide a description. Will these contributions be validated/verified? If so, please describe how. If not, why not? Is there a process for communicating/distributing these contributions to dataset consumers? If so, please provide a description.

- Yes, community contributions are encouraged and can be submitted via GitHub for validation by the UD team and/or individual treebank maintainers.

Any other comments?

- No
