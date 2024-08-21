import os


class Config:
    data_dir: str = os.path.abspath("data")
    lemmatization_dir: str = os.path.join(data_dir, "lemmatization_test")
