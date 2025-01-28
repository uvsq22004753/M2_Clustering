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


def all_cosinus_distance_matchms(spectra, tolerance=0.0):
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

    return np.array([[1.0 - scores[0] for scores in x] for x in scores])


def cosinus_binning(spec1, spec2):
    """
    Calcule la distance cosinus entre deux spectres après binning.

    Arguments
    ----------
    spec1 : Spectrum
        Premier spectre (objet Matchms) contenant les m/z et les intensités.
    spec2 : Spectrum
        Second spectre (objet Matchms) contenant les m/z et les intensités.

    Retourne
    -------
    float
        Distance cosinus entre les deux spectres.
    """
    
    spec1_mz = spec1.peaks.mz
    spec1_intensities = spec1.peaks.intensities
    spec2_mz = spec2.peaks.mz
    spec2_intensities = spec2.peaks.intensities

    i, j = 0, 0
    similarite = np.float64()

    while i < len(spec1_mz) or j < len(spec2_mz):

        if i < len(spec1_mz) and (j >= len(spec2_mz) or spec1_mz[i] < spec2_mz[j]):
            i += 1

        elif j < len(spec2_mz) and (i >= len(spec1_mz) or spec2_mz[j] < spec1_mz[i]):
            j += 1

        else:
            similarite += spec1_intensities[i] * spec2_intensities[j]
            # Avancer les indices
            i += 1
            j += 1 
    
    similarite /= (np.sum(spec1_intensities ** 2) ** 0.5 * np.sum(spec2_intensities ** 2) ** 0.5)
    
    return abs(1 - similarite)


def all_cosinus_distance_binning(spectra):
    """
    Calcule une matrice des distances cosinus entre tous les spectres d'une liste.

    Les distances sont calculées en appliquant un binning préalable sur les m/z.

    Arguments
    ----------
    spectra : list
        Liste de spectres (objets Matchms) à comparer.

    Retourne
    -------
    np.ndarray
        Matrice symétrique contenant les distances Manhattan entre chaque paire de spectres.
    """

    length = len(spectra)
    scored = np.zeros((length, length))
    for i in range(length):
        for j in range(i, length):
            score = cosinus_binning(spectra[i], spectra[j])
            # on a une matrice symétrique
            scored[i, j] = score
            scored[j, i] = score
    return scored