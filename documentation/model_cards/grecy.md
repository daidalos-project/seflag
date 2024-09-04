**greCy: grc_proiel_lg & grc_proiel_trf**

- **Person or organization developing model**: [Jacobo Myerston](https://github.com/jmyerston)

- **Model date**: 2023

- **Model version: 1.0**

- Model type: spaCy

- Information about training algorithms, parameters, fairness constraints or other applied approaches, and features: "The _lg [models] were trained using fasttext word vectors in the spaCy floret version, and the _trf models were trained using a special version of BERT, pertained by ourselves with the largest available Ancient Greek corpus, namely, the TLG. The vectors for large models were also trained with the TLG corpus." (https://github.com/jmyerston/greCy)

- Paper or other resource for more information: https://github.com/jmyerston/greCy

- License: *MIT*

- Where to send questions or comments about the model:
[jmyerston@ucsd.edu](mailto:jmyerston@ucsd.edu)

Intended Use

- Primary intended uses: Morphological analysis, POS Tagging, Lemmatization, Dependency Parsing

- Primary intended users: Classical Scholars (?)

- Out-of-scope use cases: unknown

Data, Limitations, and Recommendations

- Data selection for training: Perseus & PROIEL (UD treebanks), TLG, https://github.com/gcelano/LemmatizedAncientGreekXML

- Data selection for evaluation: Perseus & PROIEL (UD treebanks)

- Limitations: "In general, models trained with the Proiel corpus perform better in POS Tagging and Dependency Parsing, while Perseus models are better at sentence segmentation using punctuation, and Morphological Analysis. Lemmatization is similar across models because they share the same neural lemmatizer in two variants: the most accurate lemmatizer was trained with word vectors, and the other was not. The best models for lemmatization are the large models ." (https://github.com/jmyerston/greCy)
