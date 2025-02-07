import os
import numpy as np
import logging
import json
import hashlib
from datetime import datetime
from matchms.importing import load_from_mgf
from spectra.similarity.binning import fixed_binning_vector
from clustering_utilis.kmeans import normalize_features, select_best_k, run_kmeans
from clustering_utilis.common import generate_hash, write_json_results

logger = logging.getLogger(__name__)

def load_feature_matrix(mgf_file, bin_size, mz_min=20, mz_max=2000):
    """
    Charge les spectres depuis un fichier MGF et calcule leur vecteur de caractéristiques
    via fixed binning (sans normalisation, celle-ci sera appliquée par la suite).
    
    Arguments:
      - mgf_file: str, chemin vers le fichier MGF.
      - bin_size: float, largeur des bins.
      - mz_min: float, valeur minimale de m/z (défaut=20).
      - mz_max: float, valeur maximale de m/z (défaut=2000).
    
    Retourne:
      - X: matrice numpy de dimension (n_spectres, n_bins)
      - spectra_list: liste des spectres chargés (pour récupérer les métadonnées).
    """
    logger.info("Loading spectra from %s", mgf_file)
    spectra_list = list(load_from_mgf(mgf_file))
    features = [fixed_binning_vector(spec, bin_size, mz_min, mz_max) for spec in spectra_list]
    X = np.vstack(features)
    logger.info("Feature matrix shape: %s", X.shape)
    return X, spectra_list

def run_clustering_pipeline(mgf_file, bin_size, k_min, k_max, n_init=10, random_state=42,
                            algorithm='mini', mz_min=20, mz_max=2000, n_jobs=-1):
    """
    Exécute le pipeline de clustering kmeans sur un fichier MGF de spectres.
    
    Étapes :
      1. Charge le fichier MGF et calcule la matrice de caractéristiques via fixed binning.
      2. Normalise la matrice par L2.
      3. Sélectionne le meilleur nombre de clusters (k) par score de silhouette.
      4. Exécute le clustering kmeans.
      5. Génère un hash à partir des paramètres.
      6. Sauvegarde les résultats dans un fichier JSON.
    
    Retourne le chemin du fichier JSON généré.
    """
    X, spectra_list = load_feature_matrix(mgf_file, bin_size, mz_min, mz_max)
    X_norm = normalize_features(X)
    best_k, scores = select_best_k(X_norm, k_min, k_max, n_init, random_state, algorithm, n_jobs)
    labels, centers, silhouette = run_kmeans(X_norm, best_k, n_init, random_state, algorithm)
    
    # Construction des résultats
    results = []
    for i, spec in enumerate(spectra_list):
        spec_id = spec.metadata.get("id")
        results.append({"id": spec_id, "cluster": int(labels[i])})
    
    params = {
        "mgf_file": mgf_file,
        "bin_size": bin_size,
        "k_min": k_min,
        "k_max": k_max,
        "best_k": best_k,
        "n_init": n_init,
        "random_state": random_state,
        "algorithm": algorithm,
        "mz_min": mz_min,
        "mz_max": mz_max
    }
    performance = {"silhouette_score": silhouette, "scores": scores}
    
    # Générer un hash des paramètres (pour inclure une signature stable dans le nom du fichier)
    hash_val = generate_hash(params)
    
    base_name = os.path.basename(mgf_file)[:-4]  # retire l'extension .mgf
    results_dir = os.path.join("output", "clustering_results", "kmeans", "spectra", f"{base_name}_Bin{bin_size}")
    os.makedirs(results_dir, exist_ok=True)
    # Le nom final inclut le nom de base, l'algorithme, le nombre de clusters, et le hash
    output_file = os.path.join(results_dir, f"{base_name}_kmeans_{algorithm}_{best_k}_{hash_val}.json")
    
    write_json_results(params, performance, results, output_file)
    logger.info("Clustering pipeline completed. Results saved in %s", output_file)
    return output_file

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if len(sys.argv) < 6:
        print("Usage: kmeans.py <mgf_file> <bin_size> <k_min> <k_max> <algorithm> [n_init] [random_state] [mz_min] [mz_max]")
        sys.exit(1)
    mgf_file = sys.argv[1]
    bin_size = float(sys.argv[2])
    k_min = int(sys.argv[3])
    k_max = int(sys.argv[4])
    algorithm = sys.argv[5]
    n_init = int(sys.argv[6]) if len(sys.argv) > 6 else 10
    random_state = int(sys.argv[7]) if len(sys.argv) > 7 else 42
    mz_min = float(sys.argv[8]) if len(sys.argv) > 8 else 20
    mz_max = float(sys.argv[9]) if len(sys.argv) > 9 else 2000
    
    run_clustering_pipeline(mgf_file, bin_size, k_min, k_max, n_init, random_state, algorithm, mz_min, mz_max)