import os
import time
import logging
import json
import hashlib
from datetime import datetime
import numpy as np
from clustering_utilis.hac import run_hac
from clustering_utilis.common import generate_hash, write_json_results
from utils.file_utils import read_smiles_file
from smiles.similarity.matrix import generate_similarity_matrix
from smiles.similarity.representations import morgan_fingerprint
from smiles.similarity.metrics import similarity_cosinus, similarity_jaccard

logger = logging.getLogger(__name__)

def run_hac_pipeline_smiles(smiles_file: str,
                            fp_size: int = 2048,
                            k_clusters: int = 3,
                            sim_type: str = "jaccard") -> str:
    """
    Exécute le pipeline de clustering HAC sur un fichier de SMILES.
    
    Étapes :
      1. Lit le fichier de SMILES et génère la matrice de similarité via generate_similarity_matrix.
         - La transformation utilisée est morgan_fingerprint (avec fp_size).
         - La fonction de similarité choisie dépend de sim_type :
             • "cosinus" → similarity_cosinus
             • "jaccard" → similarity_jaccard
      2. Convertit la matrice de similarité en matrice de distance (distance = 1 - similarité).
      3. Applique le clustering HAC sur la matrice de distance.
      4. Génère les résultats en attribuant à chaque molécule un ID égal à son index.
      5. Génère un hash (à partir des paramètres, en excluant les données variables) et sauvegarde
         les résultats dans un fichier JSON dans le dossier :
         output/clustering_results/hac/smiles/<base_name>_fp<fp_size>/
         Le nom final du fichier JSON sera par exemple :
         "[M-3H2O+H]1+_hac_4_ab12cd34.json".
    
    Paramètres :
      - smiles_file : str  — chemin vers le fichier de SMILES (un SMILES par ligne).
      - fp_size     : int  — taille du fingerprint Morgan (défaut : 2048).
      - k_clusters  : int  — nombre de clusters à former.
      - sim_type    : str  — type de similarité ("cosinus" ou "jaccard", défaut : "jaccard").
    
    Retourne :
      - output_file : str — chemin complet du fichier JSON contenant les résultats.
    """
    # Choix de la fonction de similarité
    if sim_type.lower() == "cosinus":
        similarity_fn = similarity_cosinus
    elif sim_type.lower() == "jaccard":
        similarity_fn = similarity_jaccard
    else:
        raise ValueError("sim_type must be either 'cosinus' or 'jaccard'")
    
    transform = lambda s: morgan_fingerprint(s, fp_size=fp_size)
    
    # 1. Génération de la matrice de similarité
    sim_matrix = generate_similarity_matrix(smiles_file, transform, similarity_fn)
    logger.info("Similarity matrix generated with shape: %s", sim_matrix.shape)
    
    # 2. Conversion en matrice de distance
    distance_matrix = 1 - sim_matrix
    
    # 3. Exécution du clustering HAC sur la matrice de distance
    labels = run_hac(distance_matrix, n_clusters=k_clusters)
    
    # 4. Génération des résultats (ID = index)
    smiles_list = read_smiles_file(smiles_file)
    results = [{"id": i, "cluster": int(labels[i])} for i in range(len(smiles_list))]
    
    # 5. Préparation des paramètres et génération du hash
    params = {
        "smiles_file": smiles_file,
        "fp_size": fp_size,
        "k_clusters": k_clusters,
        "sim_type": sim_type
    }
    performance = {}  # Vous pouvez ajouter d'autres métriques si nécessaire
    
    hash_val = generate_hash(params)
    base_name = os.path.splitext(os.path.basename(smiles_file))[0]
    results_dir = os.path.join("output", "clustering_results", "hac", "smiles", f"{base_name}_fp{fp_size}")
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f"{base_name}_hac_{k_clusters}_{hash_val}.json")
    
    write_json_results(params, performance, results, output_file)
    logger.info("HAC clustering pipeline for SMILES completed. Results saved in %s", output_file)
    return output_file

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if len(sys.argv) < 4:
        print("Usage: hac_smiles.py <smiles_file> <fp_size> <k_clusters> [sim_type]")
        sys.exit(1)
    smiles_file = sys.argv[1]
    fp_size = int(sys.argv[2])
    k_clusters = int(sys.argv[3])
    sim_type = sys.argv[4] if len(sys.argv) > 4 else "jaccard"
    run_hac_pipeline_smiles(smiles_file, fp_size, k_clusters, sim_type=sim_type)
