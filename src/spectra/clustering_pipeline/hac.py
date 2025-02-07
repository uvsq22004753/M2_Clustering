import os
import time
import logging
import numpy as np
import json
import hashlib
from datetime import datetime
from spectra.similarity.binning import bin_file
from spectra.similarity.matrix import make_matrix_for_file, read_matrix
from clustering_utilis.hac import run_hac
from clustering_utilis.common import generate_hash, write_json_results
from matchms.importing import load_from_mgf

logger = logging.getLogger(__name__)

def run_hac_pipeline(mgf_file: str, bin_size: float, n_clusters: int,
                     opt: str = 'somme', mz_min: float = 20, mz_max: float = 2000,
                     tol: float = 0.1, num_workers: int = None, dist_method: str = "cosinus") -> str:
    """
    Exécute le pipeline de clustering HAC sur un fichier MGF de spectres.
    
    Étapes :
      1. Applique le binning sur le fichier MGF via la fonction bin_file.
         Le fichier binned est sauvegardé dans "output/tmp/binned_adducts_<bin_size>".
      2. Génère la matrice de distances à partir du fichier binned en utilisant
         make_matrix_for_file avec la méthode spécifiée par dist_method et tolérance tol.
         Le CSV est sauvegardé dans "output/tmp/matrix_<bin_size>_cosinus" (ou autre selon la méthode).
      3. Lit la matrice de distances (avec read_matrix) et applique HAC (run_hac) pour obtenir les labels.
      4. Recharge les spectres depuis le fichier binned pour générer les résultats,
         en utilisant les indices (IDs commençant à 1).
      5. Génère un hash à partir des paramètres (en excluant les valeurs variables) et sauvegarde
         les résultats dans un fichier JSON dans :
         "output/clustering_results/hac/spectra/<base_name>_Bin<bin_size>/".
         Le nom du fichier final intègre le nombre de clusters et le hash.
    
    Paramètres:
      - mgf_file: str           : Chemin du fichier MGF à traiter.
      - bin_size: float         : Taille du bin pour le binning.
      - n_clusters: int         : Nombre de clusters à former avec HAC.
      - opt: str                : Option pour le binning ("somme" ou "moyenne").
      - mz_min, mz_max: float    : Bornes pour le binning.
      - tol: float              : Tolérance pour le calcul de la matrice de distance.
      - num_workers: int        : Nombre de processus pour le calcul parallèle (facultatif).
      - dist_method: str        : Méthode de calcul de distance, parmi "cosinus", "manhattan", "simple", "cosine_greedy".
    
    Retourne:
      - output_file: str        : Chemin complet du fichier JSON contenant les résultats.
    """
    # 1. Binning : création du dossier temporaire pour le fichier binned
    tmp_binned_dir = os.path.join("output", "tmp", f"binned_adducts_{bin_size}")
    os.makedirs(tmp_binned_dir, exist_ok=True)
    
    # Appliquer le binning sur le fichier MGF en utilisant l'option "somme" ou "moyenne"
    binned_file = bin_file(mgf_file, tmp_binned_dir, bin_size=bin_size, opt=opt)
    logger.info("Binned file created: %s", binned_file)
    
    # 2. Génération de la matrice de distance
    tmp_matrix_dir = os.path.join("output", "tmp", f"matrix_{bin_size}_{dist_method}")
    os.makedirs(tmp_matrix_dir, exist_ok=True)
    distance_csv = make_matrix_for_file(binned_file, methode=dist_method, output_dir=tmp_matrix_dir, tol=tol, num_workers=num_workers)
    logger.info("Distance matrix CSV created: %s", distance_csv)
    
    # 3. Lecture de la matrice de distance
    distance_matrix = read_matrix(distance_csv)
    
    # 4. Exécuter le clustering HAC sur la matrice de distance
    labels = run_hac(distance_matrix, n_clusters=n_clusters)
    
    # 5. Recharger les spectres depuis le fichier binned
    spectra_list = list(load_from_mgf(binned_file))
    results = []
    for i, spec in enumerate(spectra_list):
        spec_id = spec.metadata.get("id")
        results.append({"id": spec_id, "cluster": int(labels[i])})
    
    # 6. Préparer les paramètres et générer un hash
    params = {
        "mgf_file": mgf_file,
        "bin_size": bin_size,
        "n_clusters": n_clusters,
        "opt": opt,
        "mz_min": mz_min,
        "mz_max": mz_max,
        "tol": tol,
        "dist_method": dist_method
    }
    performance = {}  # Vous pouvez ajouter d'autres métriques si nécessaire
    
    hash_val = generate_hash(params)
    base_name = os.path.splitext(os.path.basename(mgf_file))[0]
    results_dir = os.path.join("output", "clustering_results", "hac", "spectra", f"{base_name}_Bin{bin_size}")
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f"{base_name}_hac_{n_clusters}_{hash_val}.json")
    
    write_json_results(params, performance, results, output_file)
    logger.info("HAC clustering pipeline completed. Results saved in %s", output_file)
    return output_file

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if len(sys.argv) < 5:
        print("Usage: hac.py <mgf_file> <bin_size> <n_clusters> [mz_min] [mz_max] [tol] [num_workers] [dist_method]")
        sys.exit(1)
    mgf_file = sys.argv[1]
    bin_size = float(sys.argv[2])
    n_clusters = int(sys.argv[3])
    mz_min = float(sys.argv[4]) if len(sys.argv) > 4 else 20
    mz_max = float(sys.argv[5]) if len(sys.argv) > 5 else 2000
    tol = float(sys.argv[6]) if len(sys.argv) > 6 else 0.1
    num_workers = int(sys.argv[7]) if len(sys.argv) > 7 else None
    dist_method = sys.argv[8] if len(sys.argv) > 8 else "cosinus"
    run_hac_pipeline(mgf_file, bin_size, n_clusters, opt="somme", mz_min=mz_min, mz_max=mz_max, tol=tol, num_workers=num_workers, dist_method=dist_method)
