import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from rdkit import DataStructs

def similarity_cosinus(rep1, rep2) -> float:
    """
    Calcule la similarité cosinus entre deux représentations sous forme de vecteurs numpy.
    On suppose que rep1 et rep2 sont déjà des vecteurs numériques.
    """
    # On calcule la distance cosinus et on renvoie 1 - distance
    distance = cosine_distances([rep1], [rep2])[0][0]
    return 1 - distance

def similarity_jaccard(fp1, fp2) -> float:
    """
    Calcule la similarité de Jaccard (Tanimoto) entre deux fingerprints (objets RDKit).
    """
    sim = DataStructs.TanimotoSimilarity(fp1, fp2)
    return sim


def CLS(smile1: str, smile2: str) -> int:
    """
    Calcule la longueur de la plus longue sous-séquence commune (CLS) entre deux SMILES.
    """
    m, n = len(smile1), len(smile2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if smile1[i - 1] == smile2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

def CLS_distance(smile1: str, smile2: str) -> float:
    """
    Calcule une "distance" basée sur la CLS entre deux SMILES.
    On peut définir la distance comme 1 - (CLS / max(len(smile1), len(smile2))).
    """
    cls_length = 0
    # Pour éviter une division par zéro
    max_len = max(len(smile1), len(smile2))
    if max_len == 0:
        return 0.0

    cls_length = CLS(smile1, smile2)
    return 1 - (cls_length / max_len)