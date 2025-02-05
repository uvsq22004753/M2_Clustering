from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from tqdm import tqdm
import json
import os

MIN_K = 2
FP_DIR = "SMILES/data/fingerprints"
CLUSTER_DIR = "clusters/smiles"

def get_all_files_in_dir(directory):
    files = []
    for entry in os.listdir(directory):
        entry_path = os.path.join(directory, entry)
        if os.path.isfile(entry_path):
            files.append(entry_path)
    return files

def load_fp(filepath):
    fp = []
    temp = []
    with open(filepath,'r') as file:
        for line in file.readlines():
            line = line.strip()
            for i in line:
                temp.append(int(i))
            fp.append(temp)
            temp = []
    return np.array(fp)



def run_clustering(data):
    best_sil = -1
    best_labels = []
    best_k = 2
    MAX_K = 100 if len(data) > 100 else len(data)
    
    for k in tqdm(range(MIN_K, MAX_K)):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
        kmeans.fit(data)
        labels = kmeans.predict(data)
        if silhouette_score(data, labels) > best_sil:
            best_sil = silhouette_score(data, labels)
            best_labels = labels
            best_k = k
    return best_labels, best_sil, best_k
    
def build_json(labels, silhouette, k):
    json = {}
    json["k"] = k
    json["silhouette_score"] = float(silhouette)
    clusters = {}
    for i, label in enumerate(labels):
        if int(label) not in clusters:
            clusters[int(label)] = [i]
        else:
            clusters[int(label)].append(i)
    json["clusters"] = clusters
    return json

def load_in_file(json_dict, filename):
    with open(filename, 'w') as file:
        json.dump(json_dict, file, indent=4)


if __name__ == "__main__":
    FILES = get_all_files_in_dir(FP_DIR)
    for path in FILES:
        X = load_fp(path)
        best_labels, best_sil, best_k = run_clustering(X)
        json_dict = build_json(best_labels, best_sil, best_k)
        load_in_file(json_dict, f"{CLUSTER_DIR}/{path[25:-7]}_kmean.json")