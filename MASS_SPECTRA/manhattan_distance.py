import numpy as np

def manhattan_distance_binning(spec1, spec2):
    """
    Calcule la distance de Manhattan entre deux spectres après binning.

    Arguments
    ----------
    spec1 : Spectrum
        Premier spectre (objet Matchms) contenant les m/z et les intensités.
    spec2 : Spectrum
        Second spectre (objet Matchms) contenant les m/z et les intensités.

    Retourne
    -------
    float
        Distance de Manhattan totale entre les deux spectres.
    """
    
    spec1_mz = spec1.peaks.mz
    spec1_intensities = spec1.peaks.intensities
    spec2_mz = spec2.peaks.mz
    spec2_intensities = spec2.peaks.intensities

    i, j = 0, 0
    distance = 0.0

    while i < len(spec1_mz) or j < len(spec2_mz):

        if i < len(spec1_mz) and (j >= len(spec2_mz) or spec1_mz[i] < spec2_mz[j]):
            # spec1_mz[i] n'a pas de correspondance dans spec2_mz
            distance += abs(spec1_intensities[i])
            i += 1
        elif j < len(spec2_mz) and (i >= len(spec1_mz) or spec2_mz[j] < spec1_mz[i]):
            # spec2_mz[j] n'a pas de correspondance dans spec1_mz
            distance += abs(spec2_intensities[j])
            j += 1
        else:
            distance += abs(spec1_intensities[i] - spec2_intensities[j])

            # Avancer les indices
            i += 1
            j += 1 
        
    return distance


def all_manhanttan_distances_binning(spectra):
    """
    Calcule une matrice des distances Manhattan entre tous les spectres d'une liste.

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
            score = manhattan_distance_binning(spectra[i], spectra[j])
            # on a une matrice symétrique
            scored[i, j] = score
            scored[j, i] = score
    return scored



def manhattan_distance_tolerance(spec1, spec2, tolerance):
    """
    Calcule la distance de Manhattan entre deux spectres en considérant une tolérance.

    Fonctionnent sur le même principe que l'appariemment de matchms mais moins rapide que si
    on calcule une fois binning réalisé.

    Arguments
    ----------
    spec1 : Spectrum
        Premier spectre (objet Matchms) contenant les m/z et les intensités.
    spec2 : Spectrum
        Second spectre (objet Matchms) contenant les m/z et les intensités.
    tolerance : float
        Tolérance pour associer les pics m/z des deux spectres.

    Retourne
    -------
    float
        Distance de Manhattan totale entre les deux spectres.
    """
    
    spec1_mz = spec1.peaks.mz
    spec1_intensities = spec1.peaks.intensities
    spec2_mz = spec2.peaks.mz
    spec2_intensities = spec2.peaks.intensities

    # on cherche les matchs
    matches = find_matches(spec1, spec2, tolerance)
    
    score = float(0.0)
    used1 = set()
    used2 = set()

    if matches is not None:
        # paire de match avec indice mz1, indice mz2 et le resultat de la valeur absolue de la différence
        matching_pairs = matches[np.argsort(matches[:, 2], kind='mergesort')[::-1], :]
        for i in range(matching_pairs.shape[0]):
            if not matching_pairs[i, 0] in used1 and not matching_pairs[i, 1] in used2:
                score += matching_pairs[i, 2]
                # chaque pique appareillé 1 fois
                used1.add(matching_pairs[i, 0])  
                used2.add(matching_pairs[i, 1])
    
    # on rajoute les pics non utilisés encore
    unmatched_spec1 = find_unmatched_peaks(spec1_mz, {int(x) for x in used1})
    for idx in unmatched_spec1:
        score += spec1_intensities[idx]
    
    unmatched_spec2 = find_unmatched_peaks(spec2_mz, {int(x) for x in used2})
    for idx in unmatched_spec2:
        score += spec2_intensities[idx]

    return score


def find_unmatched_peaks(spec_mz, matched_indices):
    """
    Identifie les pics non appariés dans un spectre.

    Arguments
    ----------
    spec_mz : np.ndarray
        Tableau des m/z des pics dans le spectre.
    matched_indices : set
        Ensemble des indices correspondant aux pics déjà appariés.

    Retourne
    -------
    list
        Liste des indices des pics non appariés.
    """

    # Créer un ensemble complet des indices dans spec_mz
    all_indices = set(range(len(spec_mz)))

    # Identifier les indices non appariés
    unmatched_indices = all_indices - matched_indices

    return list(unmatched_indices)


def find_matches(spec1, spec2, tolerance):
    """
    Trouve les correspondances de pics entre deux spectres en utilisant une tolérance.

    Arguments
    ----------
    spec1 : Spectrum
        Premier spectre (objet Matchms) contenant les m/z et les intensités.
    spec2 : Spectrum
        Second spectre (objet Matchms) contenant les m/z et les intensités.
    tolerance : float
        Tolérance pour associer les pics m/z.

    Retourne
    -------
    np.ndarray or None
        Tableau contenant les indices des pics associés dans les deux spectres,
        ainsi que la valeur absolue de la différence des intensités.
        Retourne `None` si aucun appariement n'est trouvé.
    """

    # initialisation 
    spec1_mz = spec1.peaks.mz
    spec2_mz = spec2.peaks.mz
    spec1_intensities = spec1.peaks.intensities
    spec2_intensities = spec2.peaks.intensities

    # recherche des pics communs (on a des spectres triés)
    lowest_idx = 0
    matches = []
    for peak1_idx in range(spec1_mz.shape[0]):
        mz = spec1_mz[peak1_idx]
        low_bound = mz - tolerance
        high_bound = mz + tolerance
        for peak2_idx in range(lowest_idx, spec2_mz.shape[0]):
            mz2 = spec2_mz[peak2_idx]
            if mz2 > high_bound:
                break
            if mz2 < low_bound:
                lowest_idx = peak2_idx
            else:
                matches.append((peak1_idx, peak2_idx))
    
    idx1 = [x[0] for x in matches]
    idx2 = [x[1] for x in matches]
    
    if len(idx1) == 0:
        return None
    
    matching_pairs = []
    for i, idx in enumerate(idx1):
        matching_pairs.append([idx, idx2[i], np.abs(spec1_intensities[idx] - spec2_intensities[idx2[i]])])
    
    return np.array(matching_pairs.copy())


def all_mahattan_distances_tolerance(spectra, tolerance):
    """
    Calcule une matrice des distances Manhattan entre tous les spectres avec tolérance.

    Arguments
    ----------
    spectra : list
        Liste de spectres (objets Matchms) à comparer.
    tolerance : float
        Tolérance pour associer les pics m/z.

    Retourne
    -------
    np.ndarray
        Matrice symétrique contenant les distances Manhattan entre chaque paire de spectres.
    """

    length = len(spectra)
    scored = np.zeros((length, length))
    for i in range(length):
        for j in range(i, length):
            score = manhattan_distance_tolerance(spectra[i], spectra[j], tolerance)
            # on a une matrice symétrique
            scored[i, j] = score
            scored[j, i] = score 

    return scored