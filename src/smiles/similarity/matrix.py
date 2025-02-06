import numpy as np
from tqdm import tqdm
from utils.file_utils import read_smiles_file

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
    # Lire la liste des SMILES à partir du fichier
    smiles = read_smiles_file(smiles_file)
    
    n = len(smiles)
    sim_matrix = np.zeros((n, n))
    
    # Calcul de la matrice en ne calculant que pour i <= j (la matrice est symétrique)
    for i in tqdm(range(n), desc="Calcul de la matrice de similarité"):
        for j in range(i, n):
            if transform is not None:
                rep1 = transform(smiles[i])
                rep2 = transform(smiles[j])
                sim = similarity_fn(rep1, rep2)
            else:
                sim = similarity_fn(smiles[i], smiles[j])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim  # Symétrie
    return sim_matrix
