import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import jaccard_score
from rdkit.Chem import DataStructs

def readfile(filename):
    """
    Lit un fichier .out contenant des molécules et leurs SMILES associés, puis retourne une liste de tuples.

    Le fichier doit avoir un format où chaque ligne contient le nom d'une molécule suivi de ses SMILES,
    séparés par le caractère '\x1f'. Si une molécule a plusieurs SMILES, ceux-ci sont séparés par '\x1e'.
    La fonction traite chaque ligne du fichier, sépare les informations pertinentes, et retourne une 
    liste de tuples (nom de la molécule, liste des SMILES associés).

    Arguments:
    ----------
    filename: str 
        Le nom du fichier à lire.

    Retourne:
    ----------
    list: Une liste de tuples. Chaque tuple contient le nom d'une molécule et une liste de SMILES associés.
          Exemple: [("molécule1", ["SMILES1", "SMILES2"]), ("molécule2", ["SMILES3"])]
    """
    # Liste pour stocker les résultats
    resultats = []

    # Lecture du fichier
    with open(filename, "r") as file:
        lignes = file.readlines()

        for ligne in lignes:
            # Suppression des espaces et sauts de ligne
            ligne = ligne.strip()
            # Séparation de nom de la molécule et de ces SMILES
            nom, smiles = ligne.split('\x1f')
            # Sépare les SMILES d'une meme molécule
            smiles = smiles.split('\x1e')

            # Ajout à la liste des résultats
            resultats.append((nom, smiles))
    
    return resultats

def apply_simil_for_transform(smile1, smile2, transform, similarity_type):
    """
    Applique une fonction de similarité entre deux SMILES.

    Arguments:
    ----------
    smile1: str
        La première chaîne SMILES.  
    smile2: str
        La deuxième chaîne SMILES.  
    func: function  
        La fonction qui trnasforme les SMILES en une forme spécifique. (ex: gen_lingos, CLS, fingerprint)
    similarity_type: function
        La fonction qui calcule la similarité entre deux formes spécifiques.              

    Retourne:       
    ----------
    float: La valeur de similarité entre les deux SMILES.
    """
    return similarity_type(transform(smile1), transform(smile2))

def similarity_cosinus(smile1, smile2):
    """
    Calcule la similarité cosinus entre deux vecteurs.

    Arguments:
    ----------
    smile1: numpy.ndarray
        Le premier vecteur.
    smile2: numpy.ndarray
        Le deuxième vecteur.

    Retourne:
    ----------
    float: La similarité cosinus entre les deux vecteurs.
    """
    distance = cosine_distances([smile1], [smile2])[0][0] 

    return 1 - distance

def similarity_jaccard(smile1, smile2):
    """
    Calcule la similarité de Jaccard (Tanimoto) entre deux arrays.
    Arguments:
    ----------
    smile1: numpy.ndarray
        Le premier fingerprint.
    smile2: numpy.ndarray
        Le deuxième fingerprint.

    Retourne:
    ----------
    float: La similarité de Jaccard (Tanimoto) entre les deux arrrays.
    """
    distance = jaccard_score(smile1, smile2, average='binary')
    return 1 - distance