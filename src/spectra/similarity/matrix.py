import os
import csv
import time
import numpy as np
import multiprocessing as mp
import logging
from spectra.similarity import metrics

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

def compute_distance_matrix(file_path, methode, tol=0.1, num_workers=None):
    from matchms.importing import load_from_mgf
    logging.getLogger("matchms").setLevel(logging.ERROR)
    spectra = list(load_from_mgf(file_path))
    length = len(spectra)
    if num_workers is None:
        num_workers = mp.cpu_count()
    distance_matrix = np.zeros((length, length))
    pairs = [(i, j) for i in range(length) for j in range(i+1, length)]
    with mp.Pool(processes=num_workers) as pool:
        args = [(pair, spectra, methode, tol) for pair in pairs]
        results = pool.starmap(_compute_distance, args)
    for i, j, score in results:
        distance_matrix[j, i] = score
    return distance_matrix

def save_matrix(matrix, output_file):
    size = matrix.shape[0]
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(1, size):
            writer.writerow(matrix[i, :i])

def read_matrix(input_file):
    lower_triangular = []
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            lower_triangular.append([float(x) for x in row if x])
    size = len(lower_triangular) + 1
    square_matrix = np.zeros((size, size))
    for i in range(1, size):
        square_matrix[i, :i] = lower_triangular[i - 1]
    square_matrix += square_matrix.T
    return square_matrix

def make_matrix(input_file, methode, output_file, tol=0.1):
    deb = time.time()
    matrix_result = compute_distance_matrix(input_file, methode, tol)
    save_matrix(matrix_result, output_file)
    logging.info(f"Matrix computed and saved to {output_file} in {time.time()-deb:.2f} s.")

def make_matrices_from_dir(binned_dir, methode, output_dir, tol=0.1, num_workers=None):
    """
    Itère sur chaque fichier MGF dans binned_dir, calcule la matrice de similarité
    avec la méthode spécifiée, puis crée un sous-dossier dans output_dir pour chaque fichier
    binned dont le nom contient le nom d'adduit et la taille du bin. Le CSV est sauvegardé
    dans ce sous-dossier avec un nom incluant la méthode et (pour "simple") la tolérance.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file in os.listdir(binned_dir):
        if file.lower().endswith(".mgf"):
            input_file = os.path.join(binned_dir, file)
            base_name = file[:-4]
            subfolder = os.path.join(output_dir, base_name)
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)
            extra = f"_{methode}"
            if methode == "simple":
                extra += f"_tol{tol}"
            output_file = os.path.join(subfolder, f"{base_name}{extra}.csv")
            make_matrix(input_file, methode, output_file, tol)


def make_matrices_from_dir(binned_dir, methode, output_dir, tol=0.1, num_workers=None, bin_size=None):
    """
    Itère sur chaque fichier MGF dans binned_dir, calcule la matrice de similarité
    avec la méthode spécifiée, puis sauvegarde les CSV dans un sous-dossier de output_dir.
    
    Ce sous-dossier est nommé selon la taille de bin et la méthode, par exemple "Bin5_cosinus".
    
    Parameters:
      - binned_dir (str) : Dossier contenant les fichiers binned.
      - methode (str)    : Méthode de similarité ("cosinus", "manhattan" ou "simple").
      - output_dir (str) : Dossier racine pour les matrices de similarité.
      - tol (float)      : Tolérance (utilisée pour la méthode "simple").
      - num_workers (int): Nombre de workers pour le multiprocessing.
      - bin_size (float) : Taille du bin utilisée pour le binning.
    """
    # Définir le nom du sous-dossier avec la taille de bin et la méthode
    if bin_size is None:
        bin_size_str = "unknown"
    else:
        # Si bin_size est entier (ex: 5.0), on l'affiche sans décimales.
        bin_size_str = str(int(bin_size)) if float(bin_size).is_integer() else str(bin_size)
    subfolder_name = f"Bin{bin_size_str}_{methode}"
    subfolder_path = os.path.join(output_dir, subfolder_name)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    
    # Pour chaque fichier binned dans binned_dir, calcule la matrice et la sauvegarde dans le sous-dossier
    for file in os.listdir(binned_dir):
        if file.lower().endswith(".mgf"):
            input_file = os.path.join(binned_dir, file)
            base_name = file[:-4]  # Retire l'extension .mgf
            extra = f"_{methode}"
            if methode == "simple":
                extra += f"_tol{tol}"
            # Le nom final du CSV inclut le nom de base et la méthode, par exemple: "[M+H]1+_Bin5_cosinus.csv"
            output_file = os.path.join(subfolder_path, f"{base_name}{extra}.csv")
            make_matrix(input_file, methode, output_file, tol)
