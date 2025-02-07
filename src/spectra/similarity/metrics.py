import numpy as np

def cosinus_binning(spec1, spec2):
    """
    Calcule la distance cosinus entre deux spectres après binning.
    Retourne |1 - (cosinus des spectres)|.
    """
    spec1_mz = spec1.peaks.mz
    spec1_intensities = spec1.peaks.intensities
    spec2_mz = spec2.peaks.mz
    spec2_intensities = spec2.peaks.intensities

    i, j = 0, 0
    similarity = np.float64(0.0)
    while i < len(spec1_mz) or j < len(spec2_mz):
        if i < len(spec1_mz) and (j >= len(spec2_mz) or spec1_mz[i] < spec2_mz[j]):
            i += 1
        elif j < len(spec2_mz) and (i >= len(spec1_mz) or spec2_mz[j] < spec1_mz[i]):
            j += 1
        else:
            similarity += spec1_intensities[i] * spec2_intensities[j]
            i += 1
            j += 1 
    norm1 = np.sqrt(np.sum(spec1_intensities ** 2))
    norm2 = np.sqrt(np.sum(spec2_intensities ** 2))
    if norm1 == 0 or norm2 == 0:
        return 1.0
    similarity /= (norm1 * norm2)
    return abs(1 - similarity)

def manhattan_distance_binning(spec1, spec2):
    """
    Calcule la distance de Manhattan entre deux spectres après binning.
    """
    spec1_mz = spec1.peaks.mz
    spec1_intensities = spec1.peaks.intensities
    spec2_mz = spec2.peaks.mz
    spec2_intensities = spec2.peaks.intensities

    i, j = 0, 0
    distance = 0.0
    while i < len(spec1_mz) or j < len(spec2_mz):
        if i < len(spec1_mz) and (j >= len(spec2_mz) or spec1_mz[i] < spec2_mz[j]):
            distance += abs(spec1_intensities[i])
            i += 1
        elif j < len(spec2_mz) and (i >= len(spec1_mz) or spec2_mz[j] < spec1_mz[i]):
            distance += abs(spec2_intensities[j])
            j += 1
        else:
            distance += abs(spec1_intensities[i] - spec2_intensities[j])
            i += 1
            j += 1 
    return distance

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

def simple_similarity(spec1, spec2, tol):
    """
    Calcule une mesure de similarité simple entre deux spectres en fonction du nombre de pics m/z proches (tolérance tol).
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
    score = 1 - (2 * count) / (len(spec1_mz) + len(spec2_mz))
    return round(score, 10)

# --- Fonctions auxiliaires pour le cas tolerance ---
def find_unmatched_peaks(spec_mz, matched_indices):
    all_indices = set(range(len(spec_mz)))
    return list(all_indices - matched_indices)

def find_matches(spec1, spec2, tolerance):
    spec1_mz = spec1.peaks.mz
    spec2_mz = spec2.peaks.mz
    spec1_intensities = spec1.peaks.intensities
    spec2_intensities = spec2.peaks.intensities
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
    if not matches:
        return None
    matching_pairs = []
    for idx1, idx2 in matches:
        diff = np.abs(spec1_intensities[idx1] - spec2_intensities[idx2])
        matching_pairs.append([idx1, idx2, diff])
    return np.array(matching_pairs)
