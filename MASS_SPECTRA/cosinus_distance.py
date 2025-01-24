from matchms.importing import load_from_mgf
from matchms.similarity import CosineGreedy

import numpy as np

def list_all_spectra(file):
    """
    Charge tous les spectres depuis un fichier MGF (Mascot Generic Format).

    Paramètres :
    - file (str) : Chemin vers le fichier MGF contenant les spectres.

    Retourne :
    - list : Une liste d'objets `Spectrum` extraits du fichier.
    """
    return list(load_from_mgf(file))


def all_cosinus_similarities_matchms(spectra, tolerance=0.0):
    """
    Calcule la matrice des distances cosinus entre tous les spectres.

    Cette fonction utilise la méthode `CosineGreedy` de la bibliothèque `matchms`
    pour calculer les similarités cosinus entre chaque paire de spectres, 
    et retourne une matrice contenant ces scores.

    Paramètres :
    - spectra (list) : Liste de spectres au format accepté par `matchms`.
    - tolerance (float) : Tolérance pour correspondre aux valeurs m/z des spectres
      (défaut = 0.1).

    Retourne :
    - numpy.array : Matrice 2D des scores cosinus entre chaque paire de spectres.
    """
    
    # on initialise l'objet CosineGreedy
    cos = CosineGreedy(tolerance=tolerance)
    # on a une matrice carré symétrique
    scores = cos.matrix(spectra, spectra, is_symmetric= True)

    return np.array([[1.0 -scores[0] for scores in x] for x in scores])