from matchms.importing import load_from_mgf
from cosinus_distance import cosinus_binning
from manhattan_distance import manhattan_distance_binning
from simple_distance import simple_similarity

import time 
import logging
import shutil
import numpy as np
import csv
import os

import multiprocessing as mp

# distance cosinus
def cosinus(i, j, spectra):
    return i, j, cosinus_binning(spectra[i], spectra[j])

# distance manhattan
def manhattan(i, j, spectra):
    return i, j, manhattan_distance_binning(spectra[i], spectra[j])

# distance simple
def simple(i, j, spectra):
    return i, j, simple_similarity(spectra[i], spectra[j])


# Fonction principale pour calculer la matrice des distances efficacement
def compute_distance_matrix(file_path, methode, num_workers=None):
    """
    Calcule une matrice de distances entre tous les spectres d'un fichier MGF.

    Arguments
    ----------
    file_path : str
        Chemin du fichier MGF contenant les spectres.
    methode : str
        Méthode de comparaison ("cosinus", "manhattan" ou "simple").
    num_workers : int, optional
        Nombre de processus parallèles utilisés pour le calcul (par défaut : moitié des cœurs CPU).

    Retourne
    -------
    numpy.ndarray
        Matrice de distances triangulaire entre les spectres.
    """
    
    # initialisation
    logging.getLogger("matchms").setLevel(logging.ERROR)
    spectra = list(load_from_mgf(file_path))
    length = len(spectra)

    # choix du nombre de coeurs
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() // 2)

    # matrice des distances
    distance_matrix = np.zeros((length, length))

    # on a une matrice symétrique
    pairs = [(i, j) for i in range(length) for j in range(i+1, length)]
    
    with mp.Pool(processes=num_workers) as pool:
        
        if methode == "cosinus":
            results = pool.starmap(cosinus, [(i, j, spectra) for i, j in pairs])

        elif methode == "manhattan":
            results = pool.starmap(manhattan, [(i, j, spectra) for i, j in pairs])

        elif methode == "simple":
            results = pool.starmap(simple, [(i, j, spectra) for i, j in pairs])
        

    # Remplir la matrice avec les résultats
    for i, j, score in results:
        distance_matrix[j, i] = score 

    return distance_matrix


def save_matrix(matrix, output_file):
    """
    Sauvegarde une matrice sous forme de matrice triangulaire inférieure dans un fichier CSV.

    Arguments
    ----------
    matrix : numpy.ndarray
        Matrice de distances à enregistrer.
    output_file : str
        Chemin du fichier de sortie.

    Retourne
    -------
    None
    """
    
    size = matrix.shape[0]

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        for i in range(1, size):  # Commence à 1 car la première ligne est vide
            writer.writerow(matrix[i, :i]) 
            

def new_dir(directory):
    """
    Crée un répertoire ou supprime et recrée un répertoire existant.

    Cette fonction vérifie si le répertoire spécifié existe. Si c'est le cas, 
    elle demande à l'utilisateur s'il souhaite le supprimer. Si l'utilisateur 
    accepte, le répertoire est supprimé et recréé. Sinon, le programme s'arrête.
    Si le répertoire n'existe pas, il est simplement créé.

    Arguments
    ----------
    directory : str
        Chemin du répertoire à créer ou recréer.

    Retourne
    -------
    None
    """
    if os.path.exists(directory):
        if input(f"[WARNING] Remove '{directory}' directory ? [y/N] ").lower()=='y':
            shutil.rmtree(directory)
        else:
            print("Aborting...")
            exit(0)
    os.mkdir(directory)


def make_one_matrix(input, methode):
    """
    Génère la matrice des distances pour un fichier mgf.

    Arguments
    ----------
    input : str
        Chemin du fichier contenant les fichiers MGF.
    methode : str
        Méthode de comparaison ("cosinus", "manhattan" ou "simple").

    Retourne
    -------
    None
    """
    output = input[:-4] + "_D" + "methode" + ".csv"
    a = compute_distance_matrix(input, methode)
    save_matrix(a, output)


def make_matrix(inputdir, methode):
    """
    Génère des matrices de distances pour tous les fichiers MGF d'un dossier.

    Arguments
    ----------
    inputdir : str
        Chemin du dossier contenant les fichiers MGF.
    methode : str
        Méthode de comparaison ("cosinus", "manhattan" ou "simple").

    Retourne
    -------
    None
    """

    outputdir = inputdir + "D" + methode
    new_dir(outputdir)

    for file in os.listdir(inputdir):

            print(f"Processing {file} file...")
            
            deb = time.time()
            if not any(substr in file for substr in ["[M-H]1-", "[M+Na]1+", "[M+H]1+"]):
                input_file_path = os.path.join(inputdir, file)
                input_file_path = os.path.normpath(input_file_path)
                
                output_file = file[:-4] + "D" + methode + ".csv"
                output_file_path = os.path.join(outputdir, output_file)
                output_file_path = os.path.normpath(output_file_path)

                a = compute_distance_matrix(input_file_path, methode)
                save_matrix(a, output_file_path)

                print(f"Execution in {time.time()-deb} s.")


def read_matrix(input_file):
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
            lower_triangular.append([float(x) for x in row if x])  # Convertir en float

    # la taille de la matrice carrée
    size = len(lower_triangular) + 1  

    # Initialision
    square_matrix = np.zeros((size, size))

    # Remplir la partie triangulaire inférieure
    for i in range(1, size):  # Commence à 1 car la première ligne est vide en CSV
        square_matrix[i, :i] = lower_triangular[i - 1]  # Assigne les valeurs aux bonnes positions

    # Rendre la matrice symétrique
    square_matrix += square_matrix.T

    return square_matrix


if __name__ == "__main__":
    
    file = "./adductsBin1"
    make_matrix(file, "cosinus")
    make_matrix(file, "manhattan")
    make_matrix(file, "simple")