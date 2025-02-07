import os
import time
import logging
import json
import hashlib
from datetime import datetime
import numpy as np
from scipy.spatial.distance import pdist, squareform
from clustering_utilis.hdbscan import apply_hdbscan
from clustering_utilis.common import generate_hash, write_json_results
from utils.file_utils import read_smiles_file
from smiles.similarity.matrix import generate_similarity_matrix
from smiles.similarity.representations import morgan_fingerprint
from smiles.similarity.metrics import similarity_cosinus, similarity_jaccard

logger = logging.getLogger(__name__)

def remove_duplicate_smiles(smiles):
    """
    Élimine les doublons d'une liste de SMILES et construit un mapping qui associe chaque
    SMILES unique à la liste de ses indices dans la liste originale.
    
    Retourne:
      - unique_smiles: list of str
      - mapping: dict, clé = SMILES, valeur = liste d'indices
    """
    mapping = {}
    for i, smile in enumerate(smiles):
        if smile not in mapping:
            mapping[smile] = []
        mapping[smile].append(i)
    unique_smiles = list(mapping.keys())
    return unique_smiles, mapping

def map_labels(mapping, unique_labels, total):
    """
    Affecte à chaque SMILES de la liste originale le label correspondant,
    en se basant sur le mapping des SMILES uniques.
    Retourne:
      - list of int: liste des labels assignés, dans l'ordre de l'ensemble original.
    """
    res = [None] * total
    for idx, smile in enumerate(mapping.keys()):
        label = unique_labels[idx]
        for i in mapping[smile]:
            res[i] = label
    return res

def run_hdbscan_pipeline_smiles(smiles_file: str,
                                fp_size: int = 2048,
                                min_cluster_size: int = 4,
                                min_samples: int = 1,
                                sim_type: str = "jaccard") -> str:
    """
    Exécute le pipeline de clustering HDBSCAN sur un fichier de SMILES.
    
    Étapes :
      1. Lit le fichier de SMILES et élimine les doublons pour obtenir une liste unique et un mapping.
      2. Crée un fichier temporaire contenant uniquement les SMILES uniques.
      3. Utilise la fonction generate_similarity_matrix sur ce fichier pour générer la matrice de similarité.
      4. Convertit la matrice de similarité en matrice de distance : distance = 1 - similarité.
      5. Applique HDBSCAN sur la matrice de distance pour obtenir les labels pour les SMILES uniques.
      6. Réaffecte ces labels aux SMILES originaux via le mapping.
      7. Génère un hash à partir des paramètres et sauvegarde les résultats dans un fichier JSON dans :
         output/clustering_results/hdbscan/smiles/<base_name>_fp<fp_size>/
         Le nom du fichier final inclut min_cluster_size et le hash.
    
    Paramètres:
      - smiles_file : str   : Chemin du fichier de SMILES (un SMILES par ligne).
      - fp_size     : int   : Taille du fingerprint Morgan (défaut : 2048).
      - min_cluster_size : int : Taille minimale d'un cluster pour HDBSCAN.
      - min_samples : int   : Nombre minimum d'échantillons pour HDBSCAN (défaut : 1).
      - sim_type    : str   : Type de similarité à utiliser ("cosinus" ou "jaccard", défaut : "jaccard").
    
    Retourne:
      - output_file : str  : Chemin complet du fichier JSON contenant les résultats.
    
    Remarque : Les IDs dans les résultats correspondent aux indices originaux dans le fichier (commençant à 0).
    """
    # 1. Lire le fichier de SMILES
    original_smiles = read_smiles_file(smiles_file)
    total = len(original_smiles)
    unique_smiles, mapping = remove_duplicate_smiles(original_smiles)
    logger.info("Found %d unique SMILES out of %d", len(unique_smiles), total)
    
    # 2. Écrire les SMILES uniques dans un fichier temporaire
    base_name = os.path.splitext(os.path.basename(smiles_file))[0]
    temp_dir = os.path.join("output", "tmp", "unique_smiles")
    os.makedirs(temp_dir, exist_ok=True)
    unique_file = os.path.join(temp_dir, f"{base_name}_unique.smiles")
    with open(unique_file, "w") as f:
        for smile in unique_smiles:
            f.write(smile + "\n")
    
    # 3. Choix de la fonction de similarité
    if sim_type.lower() == "cosinus":
        similarity_fn = similarity_cosinus
    elif sim_type.lower() == "jaccard":
        similarity_fn = similarity_jaccard
    else:
        raise ValueError("sim_type must be either 'cosinus' or 'jaccard'")
    
    # Transformation : morgan_fingerprint avec fp_size
    transform = lambda s: morgan_fingerprint(s, fp_size=fp_size)
    
    # 4. Générer la matrice de similarité à partir du fichier unique
    sim_matrix = generate_similarity_matrix(unique_file, transform, similarity_fn)
    logger.info("Similarity matrix generated with shape: %s", sim_matrix.shape)
    
    # 5. Conversion en matrice de distance
    distance_matrix = 1 - sim_matrix
    
    # 6. Appliquer HDBSCAN sur la matrice de distance pour les SMILES uniques
    labels_unique, max_label = apply_hdbscan(distance_matrix, min_cluster_size, min_samples)
    
    # 7. Réaffecter les labels aux SMILES originaux via le mapping
    mapped_labels = map_labels(mapping, labels_unique, total)
    
    # 8. Préparer les résultats (IDs = indices originaux, commençant à 0)
    results = [{"id": i, "cluster": int(mapped_labels[i])} for i in range(total)]
    
    params = {
        "smiles_file": smiles_file,
        "fp_size": fp_size,
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "sim_type": sim_type
    }
    performance = {"max_label": max_label}
    
    hash_val = generate_hash(params)
    results_dir = os.path.join("output", "clustering_results", "hdbscan", "smiles", f"{base_name}_fp{fp_size}")
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f"{base_name}_hdbscan_{min_cluster_size}_{hash_val}.json")
    
    write_json_results(params, performance, results, output_file)
    logger.info("HDBSCAN clustering pipeline for SMILES completed. Results saved in %s", output_file)
    return output_file

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if len(sys.argv) < 4:
        print("Usage: hdbscan_smiles.py <smiles_file> <fp_size> <min_cluster_size> [min_samples] [sim_type]")
        sys.exit(1)
    smiles_file = sys.argv[1]
    fp_size = int(sys.argv[2])
    min_cluster_size = int(sys.argv[3])
    min_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    sim_type = sys.argv[5] if len(sys.argv) > 5 else "jaccard"
    run_hdbscan_pipeline_smiles(smiles_file, fp_size, min_cluster_size, min_samples, sim_type=sim_type)
