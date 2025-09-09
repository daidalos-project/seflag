import os
import csv
import requests
import gensim
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import pandas as pd
from word_embeddings.eval_functions import evaluate_grc_embeddings, evaluate_lat_embeddings

def run_evaluation(metafile, goldpath, gold_language="Greek", goldtype="similarity", file_ending=".bin", model_type="fasttext"):  
    results = []

    with open(metafile, 'r', encoding="utf-8") as mfile:
        reader = csv.DictReader(mfile, delimiter='\t')
        for row in reader:
            model_name = row["model_name"]
            data_format = row["data_format"]
            language = row["language"]
            period = row["period"]
            embedding_method = row["embedding_method"]
            download_link = row["download_link"]

            print("Row:", row)
            print("Checking:", data_format, file_ending, embedding_method, model_type, language, gold_language)

            if data_format.strip().lstrip(".").lower() == file_ending.strip().lstrip(".").lower() \
            and embedding_method.strip().lower() == model_type.lower() \
            and language.strip().lower() == gold_language.lower():
                model_path = os.path.join(os.getcwd(), f"word_embeddings/models/{model_name}{data_format}")
                if not os.path.exists(model_path):
                    print(f"  Downloading {model_name}...")
                    r = requests.get(download_link, stream=True)
                    with open(model_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                print(f"Loading {model_name}...")
                if embedding_method == "fasttext":
                    model = gensim.models.fasttext.load_facebook_model(model_path)
                if embedding_method == "word2Vec" \
                and data_format == ".bin":
                    model = KeyedVectors.load_word2vec_format(datapath(model_path), binary=True)
                # insert code for other types like ELMo
                print(f"Evaluating {model_name}...")
                if language == "Greek":
                    if goldtype == "similarity":
                        pearson, spearman, oov_ratio, top_1_accuracy, top_10_recall = evaluate_grc_embeddings(model, goldpath)
                        results.append({
                            "model_name": model_name,
                            "language": language,
                            "period": period,
                            "embedding_method": embedding_method,
                            "gold_standard": os.path.basename(goldpath),
                            "top_1_accuracy": top_1_accuracy,
                            "top_10_recall": top_10_recall,
                            "spearman_corr": spearman,
                            "pearson's r": pearson,
                            "oov_ratio": oov_ratio,
                            })
                    # add code for different goldtypes
                    # if goldtype == "analogy":
                    # if goldtype == "synonymity":
                if language == "Latin":
                    # add code for different goldtypes
                    # if goldtype == "similarity":
                    # if goldtype == "analogy":
                    if goldtype == "synonymity":
                        distractor_ranking, top_1_accuracy, top_10_recall = evaluate_lat_embeddings(model, goldpath)
                        results.append({
                            "model_name": model_name,
                            "language": language,
                            "period": period,
                            "embedding_method": embedding_method,
                            "top_1_accuracy": top_1_accuracy,
                            "top_10_recall": top_10_recall,
                            "distractor_ranking": distractor_ranking
                            })
    df = pd.DataFrame(results)
    print(df)
    df.to_csv("evaluation_results.csv", index=False)
    print("Done.")

metafile = "C:/Users/IuvenisGrammaticus/Documents/Python_Arbeit/word_embeddings/Modell_Tabelle.csv"
goldpath = "C:/Users/IuvenisGrammaticus/Documents/Python_Arbeit/data/word-embeddings-dicts/AGREE/2_agree_task2.json"

run_evaluation(metafile, goldpath)


# Top-1 Accuracy und top-10 recall umbenennen in accuracy