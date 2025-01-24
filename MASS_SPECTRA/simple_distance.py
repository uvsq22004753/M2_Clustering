import numpy as np

def simple_similarity(spec1, spec2, tol):
    """
    Calcule une mesure de similarité simple entre deux spectres en fonction des valeurs m/z.

    Cette fonction détermine le nombre de pics m/z "proches" entre deux spectres, 
    en tenant compte d'une tolérance définie. Elle renvoie une valeur de similarité
    entre 0 et 1 basée sur le nombre de correspondances.

    Paramètres :
    ----------
    - spec1 : Spectrum
        Premier spectre contenant les valeurs m/z à comparer.
    - spec2 : Spectrum
        Second spectre contenant les valeurs m/z à comparer.
    - tol : float, optional
        Tolérance pour considérer deux pics m/z comme "proches" (défaut = 0).

    Retourne :
    -------
    - float
        Score de similarité 
    """
    
    spec1_mz = spec1.peaks.mz
    spec2_mz = spec2.peaks.mz

    i, j, count = 0, 0, 0
    while i < len(spec1_mz) and j < len(spec2_mz):
        if abs(spec1_mz[i] - spec2_mz[j]) <= tol:
            count += 1
            i += 1
            j += 1
        elif spec1_mz[i] < spec2_mz[j]:
            i += 1
        else:
            j += 1
    
    print(count)
    
    return 1 - (2*count)/(len(spec1_mz)+len(spec2_mz))


def all_simple_similarities(spectra, tolerance=0.0):
    """
    Calcule une matrice de similarité pour une liste de spectres, 
    en utilisant la fonction `simple_similarity`.

    Cette fonction compare chaque paire de spectres dans la liste 
    et génère une matrice symétrique.

    Paramètres :
    ----------
    - spectra : list
        Liste de spectres, chaque spectre devant contenir les valeurs m/z dans l'attribut `peaks.mz`.
    - tolerance : float, optional
        Tolérance pour considérer deux pics m/z comme similaires (défaut = 0.1).

    Retourne :
    -------
    - numpy.ndarray
        Matrice 2D symétrique contenant les scores de similarité entre tous les spectres.
    """

    length = len(spectra)
    scored = np.zeros((length, length))
    for i in range(length):
        for j in range(i, length):
            score = simple_similarity(spectra[i], spectra[j], tolerance)
            # on a une matrice symétrique
            scored[i, j] = score
            scored[j, i] = score 

    return scored