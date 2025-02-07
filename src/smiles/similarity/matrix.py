import numpy as np
from utils.file_utils import read_smiles_file
from rdkit import DataStructs
from smiles.similarity.representations import morgan_fingerprint

def generate_similarity_matrix(smiles_file: str, transform, similarity_fn) -> np.ndarray:
    """
    Lit un fichier contenant des SMILES (un SMILES par ligne) et génère la matrice de similarité.
    
    Arguments:
      - smiles_file : str
          Chemin vers le fichier texte contenant les SMILES.
      - transform : function
          Fonction qui transforme un SMILES en une représentation (ex: fingerprint, n‑grammes, etc.).
          Si transform est None, la fonction de similarité sera appliquée directement sur les SMILES.
      - similarity_fn : function
          Fonction qui calcule la similarité entre deux représentations.
    
    Retourne:
      - np.ndarray : La matrice de similarité (de dimension n x n, où n est le nombre de SMILES).
    """
    smiles = read_smiles_file(smiles_file)
    n = len(smiles)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if transform is not None:
                rep1 = transform(smiles[i])
                rep2 = transform(smiles[j])
                sim = similarity_fn(rep1, rep2)
            else:
                sim = similarity_fn(smiles[i], smiles[j])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim  # Assurer la symétrie
    return sim_matrix

def generate_fingerprint_matrix(smiles_file: str, fp_size: int = 2048) -> np.ndarray:
    """
    Lit un fichier contenant des SMILES (un SMILES par ligne) et génère une matrice de fingerprints.
    
    Chaque SMILES est transformé en fingerprint (par défaut de taille fp_size via la méthode Morgan),
    puis converti en un vecteur numpy binaire.
    
    Arguments:
      - smiles_file : str
          Chemin vers le fichier texte contenant les SMILES.
      - fp_size : int, optionnel
          Taille du fingerprint à générer (défaut 2048).
    
    Retourne:
      - np.ndarray : Une matrice 2D de dimension (n_molecules, fp_size), 
                     où chaque ligne correspond au fingerprint binaire d’un SMILES.
    """
    smiles = read_smiles_file(smiles_file)
    n = len(smiles)
    fp_matrix = np.zeros((n, fp_size), dtype=int)
    for i, smile in enumerate(smiles):
        try:
            fp = morgan_fingerprint(smile, fp_size)
        except Exception as e:
            raise ValueError(f"Erreur lors de la génération du fingerprint pour SMILES '{smile}': {e}")
        arr = np.zeros((fp_size,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fp_matrix[i, :] = arr
    return fp_matrix
