from fingerprints import *
from CLS import *
from lingo import *
from smiles_utils import *
from tqdm import tqdm

def gen_all_dist(smiles, transform, similarity_type):
    """
    Génère une matrice de similarité entre les SMILES.
    
    Arguments:
    ----------
    smiles: list
        La liste des SMILES.
    transform: function
        La fonction qui transforme les SMILES en une forme spécifique. (ex: gen_lingos, CLS, fingerprint)
    similarity_type: function
        La fonction qui calcule la similarité entre deux formes spécifiques.              
    
    Retourne:
    ----------
    numpy.ndarray: La matrice de similarité entre les SMILES.
    """
    resultats = []
    for i in tqdm(range(len(smiles))):
        valeurs = []
        for j in range(i, len(smiles)):
            if transform != None:
                valeurs.append(apply_simil_for_transform(smiles[i], smiles[j], transform, similarity_type))
            else:
                valeurs.append(similarity_type(smiles[i], smiles[j]))
        resultats.append(valeurs)

    return resultats

def write_matrix(matrix, filename):
    """
    Écrit une matrice dans un fichier texte.
    
    Arguments:
    ----------
    matrix: numpy.ndarray
        La matrice à écrire.
    filename: str
        Le nom du fichier texte.
    """
    with open(filename, "x") as file:
        for row in matrix:
            for val in row[:-1]:
                file.write(f"{val},")
            file.write(f"{row[-1]}")
            file.write("\n")


if __name__ == "__main__":
    # Génération de la matrice de similarité entre les SMILES
    smiles = readfile_without_cn("SMILES/smiles_without_cn.txt")

    #mat_lingo_cos = gen_all_dist(smiles, gen_lingos, similarity_cosinus)
    #write_matrix(mat_lingo_cos, "SMILES/matrix_lingo_cos.txt")

    #mat_lingo_jacc = gen_all_dist(smiles, gen_lingos, similarity_jaccard)
    #write_matrix(mat_lingo_jacc, "SMILES/matrix_lingo_jacc.txt")

    mat_CLS = gen_all_dist(smiles, None, CLS)
    write_matrix(mat_CLS, "SMILES/matrix_CLS.txt")

    mat_fingerprint_jacc = gen_all_dist(smiles, fingerprint_morgan, similarity_jaccard)
    write_matrix(mat_fingerprint_jacc, "SMILES/matrix_fingerprint_jacc.txt")
    mat_fingerprint_cos = gen_all_dist(smiles, fingerprint_morgan, similarity_cosinus)
    write_matrix(mat_fingerprint_cos, "SMILES/matrix_fingerprint_cos.txt")