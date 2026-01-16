'''IMPORT NEEDED MODULES''' 

# Path handling #
import os

# Time Management #
import tqdm as tqdm

# Processing and Reading files #
import json
import csv

def evaluate_grc_embeddings(model, goldpath):
    '''EVALUATION'''
    print("Exāmināre incipiō.")
    print("Compōnō mēnsūra comparātiōnis.\n(anglicē: Setting up gold standard.)")

    with open(goldpath, 'r') as file:
        gold_data = json.load(file)["pairs"]

    print("Imprimō īnspectiōnis causā mēnsūrae comparātiōnis fragmentum prīmum.\n(anglicē: Printing first entry in gold_data for inspection.)")
    print(gold_data[:1])

    print("Mēnsūra comparātiōnis in virgulā intercīsam fōrmam vertō.\n(anglicē: Setting up gold standard .csv file.)")

    grc_pairs_and_scores_list = []

    for entry in gold_data:
        removed_comma = entry["pair"].replace(",","")
        pair = removed_comma.strip().split()
        w1 = pair[0]
        w2 = pair[1]
        score = entry["score"]
        grc_pairs_and_scores_list.append([w1,w2,score])

    csv_file_path = os.path.join(os.getcwd(), "data/word-embeddings-dicts/AGREE/similarity_scores.csv")

    with open(csv_file_path, "w", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerows(grc_pairs_and_scores_list)

    print("Similitūdō cosinūs, summum vocabulum, decem summa vocabula computō.\n(anglicē:Computing cosine similarity, top 1 and top 10 most similar words.)")

    is_evaluable_counter = 0
    is_not_evaluable_counter = 0

    gold_scores_list = []
    cos_similarity_list = []
    top1_hits = 0
    top10_hits = 0 

    for entry in gold_data:   
        
        # Prepare word pairs
        removed_comma = entry["pair"].replace(",","")
        pair = removed_comma.strip().split()
        w1 = pair[0]
        w2 = pair[1]
        score = entry["score"]
        
        # Check if words are in model vocab
        if w1 not in model.wv:
            is_not_evaluable_counter += 1
            print(f"{w1} in {entry} not in model vocab")
            continue
        if  w2 not in model.wv:
            is_not_evaluable_counter += 1
            print(f"{w2} in {entry} not in model vocab")
            continue
        if w1 in model.wv and w2 in model.wv:
            is_evaluable_counter += 1

            # Save gold scores
            gold_scores_list.append(score)

            # Compute cosine similarity
            # cos_similarity = model.wv.similarity(w1,w2)
            # cos_similarity_list.append(cos_similarity)

            # Compute top 1 most similar word
            top_1 = model.wv.most_similar(positive=w1, topn=1)[0][0]
                    # returns list of tupels for each of the topn words. 
                    # We need only the word, i.e. the first element of the first tupel in the list.
            if top_1 == w2:
                top1_hits += 1
        
            # Compute top 10 most similar word
            top_10_list = []
            top_10 = model.wv.most_similar(positive=w1, topn=10)
            for x in top_10:
                top_10_list.append(x[0])
            if w2 in top_10_list:
                top10_hits += 1

    # Compute Pearson's r, Spearman's r and out-of-vocabulary-ratio
    pearson, spearman, oov_ratio = model.wv.evaluate_word_pairs(pairs=csv_file_path, delimiter=',')
    evaluated_pairs = is_evaluable_counter / (is_evaluable_counter + is_not_evaluable_counter)
    top_1_accuracy = top1_hits / is_evaluable_counter
    top_10_recall = top10_hits / is_evaluable_counter

    return pearson, spearman, oov_ratio, top_1_accuracy, top_10_recall

def evaluate_lat_embeddings(model, goldpath):
        
    '''EVALUATION'''

    print("Exāmināre incipiō.")
    print("Compōnō mēnsūra comparātiōnis.\n(anglicē: Setting up gold standard.)")

    gold_data = []

    with open(goldpath, 'r', encoding="utf-8") as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        for row in tsv_reader:
            gold_data.append(row)

    print("Imprimō īnspectiōnis causā mēnsūrae comparātiōnis fragmentum prīmum.\n(anglicē: Printing first entry in gold_data for inspection.)")

    print(gold_data[0])

    print("Similitūdō cosinūs, summum vocabulum, decem summa vocabula computō.\n(anglicē:Computing cosine similarity, top 1 and top 10 most similar words.)")

    is_evaluable_counter = 0
    is_not_evaluable_counter = 0

    syn_higher_than_distr_counter = 0
    top1_hits = 0
    top10_hits = 0 

    for entry in gold_data:   
        
        # Prepare word pairs
        w1 = entry[0]
        w2 = entry[1]
        distractors = entry[2:]
        
        # Check if words are in model vocab
        if w1 not in model.wv:
            is_not_evaluable_counter += 1
            print(f"{w1} in {entry} not in model vocab")
            continue
        if  w2 not in model.wv:
            is_not_evaluable_counter += 1
            print(f"{w2} in {entry} not in model vocab")
            continue
        if w1 in model.wv and w2 in model.wv:
            is_evaluable_counter += 1

            # Compute distractor ranking
            syn_similarity = model.wv.similarity(w1,w2)
            distractor_sims = []
            for d in distractors:
                if d in model.wv:
                    distractor_sims.append(model.wv.similarity(w1, d))
            if all(syn_similarity > ds for ds in distractor_sims):
                syn_higher_than_distr_counter += 1
            
            # Compute top 1 most similar word
            top_1 = model.wv.most_similar(positive=w1, topn=1)[0][0]
                    # returns list of tupels for each of the topn words. 
                    # We need only the word, i.e. the first element of the first tupel in the list.
            if top_1 == w2:
                top1_hits += 1
        
            # Compute top 10 most similar word
            top_10_list = []
            top_10 = model.wv.most_similar(positive=w1, topn=10)
            for x in top_10:
                top_10_list.append(x[0])
            if w2 in top_10_list:
                top10_hits += 1
    
    evaluated_pairs = is_evaluable_counter / (is_evaluable_counter + is_not_evaluable_counter)
    top_1_accuracy = top1_hits / is_evaluable_counter
    top_10_recall = top10_hits / is_evaluable_counter
    distractor_ranking = syn_higher_than_distr_counter / is_evaluable_counter

    return distractor_ranking, top_1_accuracy, top_10_recall