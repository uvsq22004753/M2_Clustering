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
    
    return round(1 - (2*count)/(len(spec1_mz)+len(spec2_mz)), 10)