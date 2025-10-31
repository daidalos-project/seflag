import os


class Config:
    data_dir: str = os.path.abspath("data")
    lemmatization_dir: str = os.path.join(data_dir, "Lemmatisation")
    dependency_dir: str = os.path.join(data_dir, "Dependency_Parsing")
    pos_dir: str = os.path.join(data_dir, "POS_Tagging")
    ner_dir: str = os.path.join(data_dir, "NER")
    wordemb_dir: str = os.path.join(data_dir, "Word_Embeddings")
