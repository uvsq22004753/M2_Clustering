import numpy as np

def gen_lingos(smile, n):
    """
    Génère les n-grammes de caractères (lingos) d'une SMILES.

    Arguments
    ----------
    smile : str
        La SMILES à traiter.
    n : int
        La taille des n-grammes.

    Retourne:
    -------
    lingo_list : list
        La liste des n-grammes générés.
    """
    
    lingo_list = [smile[i:i+n] for i in range(len(smile)-n+1)]
    
    return np.array(lingo_list)