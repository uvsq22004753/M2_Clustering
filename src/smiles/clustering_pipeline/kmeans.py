import os
import numpy as np
import logging
import json
import hashlib
from datetime import datetime
from utils.file_utils import read_smiles_file
from smiles.similarity.matrix import generate_fingerprint_matrix
from clustering_utilis.kmeans import normalize_features, select_best_k, run_kmeans, write_json_results

logger = logging.getLogger(__name__)

def load_feature_matrix(smiles_file: str, fp_size: int = 2048):
    """
    Lit un fichier contenant des SMILES (un SMILES par ligne) et génère la matrice de fingerprints.
    
    Retourne:
      - X : matrice numpy de dimension (n_samples, fp_size)
      - smiles_list : liste des SMILES (pour les métadonnées)
    """
    smiles_list = read_smiles_file(smiles_file)
    X = generate_fingerprint_matrix(smiles_file, fp_size=fp_size)
    logger.info("Fingerprint matrix shape: %s", X.shape)
    return X, smiles_list

def generate_hash(metadata: dict) -> str:
    """
    Génère un hash MD5 à partir d'un dictionnaire de paramètres, en excluant les clés variables (date, timestamp).
    Retourne une chaîne de 8 caractères.
    """
    filtered = {k: metadata[k] for k in sorted(metadata) if k not in ['date', 'timestamp']}
    metadata_str = json.dumps(filtered, sort_keys=True)
    return hashlib.md5(metadata_str.encode('utf-8')).hexdigest()[:8]

def run_clustering_pipeline(smiles_file: str, fp_size: int, k_min: int, k_max: int,
                            n_init: int = 10, random_state: int = 42, algorithm: str = 'mini',
                            n_jobs: int = -1) -> str:
    """
    Exécute le pipeline de clustering kmeans sur un fichier de SMILES.
    
    Étapes :
      1. Lit le fichier de SMILES et génère la matrice de fingerprints via generate_fingerprint_matrix.
      2. Normalise la matrice par L2.
      3. Sélectionne le meilleur nombre de clusters (k) via le score de silhouette.
      4. Exécute le clustering kmeans.
      5. Génère un hash à partir des paramètres.
      6. Sauvegarde les résultats dans un fichier JSON dans output/clustering_results/kmeans/smiles.
    
    Retourne:
      - Le chemin complet du fichier JSON généré.
    """
    X, smiles_list = load_feature_matrix(smiles_file, fp_size)
    X_norm = normalize_features(X)
    best_k, scores = select_best_k(X_norm, k_min, k_max, n_init, random_state, algorithm, n_jobs)
    labels, centers, silhouette = run_kmeans(X_norm, best_k, n_init, random_state, algorithm)
    
    # Préparation des résultats : associer chaque SMILES à son cluster
    results = []
    for i, smile in enumerate(smiles_list):
        results.append({"smile": smile, "cluster": int(labels[i])})
    
    params = {
        "smiles_file": smiles_file,
        "fp_size": fp_size,
        "k_min": k_min,
        "k_max": k_max,
        "best_k": best_k,
        "n_init": n_init,
        "random_state": random_state,
        "algorithm": algorithm
    }
    performance = {"silhouette_score": silhouette, "scores": scores}
    
    hash_val = generate_hash(params)
    base_name = os.path.splitext(os.path.basename(smiles_file))[0]
    results_dir = os.path.join("output", "clustering_results", "kmeans", "smiles", f"{base_name}_fp{fp_size}")
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f"{base_name}_kmeans_{algorithm}_{best_k}_{hash_val}.json")
    
    write_json_results(params, performance, results, output_file)
    logger.info("Clustering pipeline for SMILES completed. Results saved in %s", output_file)
    return output_file

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if len(sys.argv) < 6:
        print("Usage: kmeans_smiles.py <smiles_file> <fp_size> <k_min> <k_max> <algorithm> [n_init] [random_state]")
        sys.exit(1)
    smiles_file = sys.argv[1]
    fp_size = int(sys.argv[2])
    k_min = int(sys.argv[3])
    k_max = int(sys.argv[4])
    algorithm = sys.argv[5]
    n_init = int(sys.argv[6]) if len(sys.argv) > 6 else 10
    random_state = int(sys.argv[7]) if len(sys.argv) > 7 else 42
    run_clustering_pipeline(smiles_file, fp_size, k_min, k_max, n_init, random_state, algorithm)
