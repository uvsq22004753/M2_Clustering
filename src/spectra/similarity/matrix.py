import os
import csv
import time
import numpy as np
import multiprocessing as mp
import logging
from matchms.importing import load_from_mgf
from utils.file_utils import new_dir
from spectra.similarity import metrics  # Assurez-vous que ce chemin est correct selon votre structure

def _compute_distance(pair, spectra, methode, tol):
    i, j = pair
    if methode == "cosinus":
        score = metrics.cosinus_binning(spectra[i], spectra[j])
    elif methode == "manhattan":
        score = metrics.manhattan_distance_binning(spectra[i], spectra[j])
    elif methode == "simple":
        score = metrics.simple_similarity(spectra[i], spectra[j], tol)
    else:
        raise ValueError(f"Méthode inconnue: {methode}")
    return i, j, score

def compute_distance_matrix(file_path: str, methode: str, tol: float = 0.1, num_workers: int = None) -> np.ndarray:
    """
    Calcule la matrice de distance pour le fichier MGF spécifié en utilisant la méthode indiquée.
    
    Si la méthode est "cosine_greedy", on utilise l'objet CosineGreedy de matchms pour
    calculer la matrice en une seule passe.
    """
    logging.getLogger("matchms").setLevel(logging.ERROR)
    spectra = list(load_from_mgf(file_path))
    length = len(spectra)
    
    if methode == "cosine_greedy":
        from matchms.similarity import CosineGreedy
        cos = CosineGreedy(tolerance=tol)
        scores = cos.matrix(spectra, spectra, is_symmetric=True)
        # Convertit les similarités en distances : distance = 1 - similarité
        distance_matrix = np.array([[1.0 - s for s in row] for row in scores])
        return distance_matrix

    if num_workers is None:
        num_workers = mp.cpu_count()  # Ou mp.cpu_count()-1 pour laisser une marge

    distance_matrix = np.zeros((length, length))
    pairs = [(i, j) for i in range(length) for j in range(i+1, length)]
    with mp.Pool(processes=num_workers) as pool:
        args = [(pair, spectra, methode, tol) for pair in pairs]
        results = pool.starmap(_compute_distance, args)
    for i, j, score in results:
        distance_matrix[j, i] = score
    return distance_matrix

def save_matrix(matrix: np.ndarray, output_file: str):
    size = matrix.shape[0]
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(1, size):
            writer.writerow(matrix[i, :i])

def read_matrix(input_file: str) -> np.ndarray:
    """
    Lit une matrice triangulaire inférieure depuis un fichier CSV et la reconstruit en matrice carrée.

    Arguments
    ----------
    input_file : str
        Chemin du fichier CSV contenant la matrice.

    Retourne
    -------
    numpy.ndarray
        Matrice symétrique de distances.
    """
    lower_triangular = []
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            lower_triangular.append([float(x) for x in row if x])
    # La taille de la matrice carrée est le nombre de lignes + 1
    size = len(lower_triangular) + 1  
    square_matrix = np.zeros((size, size))
    for i in range(1, size):
        square_matrix[i, :i] = lower_triangular[i - 1]
    square_matrix += square_matrix.T
    return square_matrix

def make_matrix_for_file(input_file: str, methode: str, output_dir: str, tol: float = 0.1, num_workers: int = None) -> str:
    """
    Calcule la matrice de distance pour un fichier MGF binned et sauvegarde le résultat dans un sous-dossier.
    
    Le sous-dossier est créé dans output_dir et porte le nom de base du fichier binned.
    Le nom du fichier CSV intègre la méthode utilisée (et la tolérance, le cas échéant).
    
    Retourne le chemin complet du CSV généré.
    """
    deb = time.time()
    matrix_result = compute_distance_matrix(input_file, methode, tol, num_workers)
    base_name = os.path.basename(input_file)[:-4]  # Retire l'extension .mgf
    subfolder = os.path.join(output_dir, base_name)
    new_dir(subfolder)
    if methode == "cosine_greedy":
        extra = "_cosine_greedy"
    else:
        extra = f"_{methode}"
        if methode == "simple":
            extra += f"_tol{tol}"
    output_file = os.path.join(subfolder, f"{base_name}{extra}.csv")
    save_matrix(matrix_result, output_file)
    logging.info(f"Matrix computed and saved to {output_file} in {time.time()-deb:.2f} s.")
    return output_file
