import numpy as np
import logging

def clean_compound_name(compound_name: str) -> str:
    """
    Retourne le nom de la molécule sans le champ 'CollisionEnergy'.
    """
    if "CollisionEnergy" in compound_name:
        tmp = compound_name.rsplit("CollisionEnergy", 1)
        if len(tmp) >= 3:
            logging.warning("Multiple 'CollisionEnergy' occurrences found in compound name.")
        return tmp[0]
    return compound_name

def filter_params(params: dict) -> dict:
    """
    Extrait les paramètres pertinents et nettoie le nom de la molécule.
    """
    compound_name = params.get('compound_name', '')
    compound_name = clean_compound_name(compound_name)
    smiles = params.get('smiles', '')
    return {"compound_name": compound_name, "smiles": smiles}

def filter_peaks(mz_array, intensity_array, mz_from: float, mz_to: float, min_intensity: float):
    """
    Filtre et normalise les pics.
    - Conserve les pics dont la valeur m/z est comprise entre mz_from et mz_to.
    - Normalise l'intensité et supprime les pics dont l'intensité est inférieure à min_intensity.
    """
    mask = np.logical_and(mz_array >= mz_from, mz_array <= mz_to)
    mz_array = mz_array[mask]
    intensity_array = intensity_array[mask]

    if intensity_array.size == 0:
        return np.empty(0), np.empty(0)

    max_intensity = np.max(intensity_array)
    if max_intensity <= 0:
        return np.empty(0), np.empty(0)

    # Normalisation
    intensity_array = intensity_array / max_intensity

    mask = intensity_array > min_intensity
    return mz_array[mask], intensity_array[mask]

def fingerprint(params: dict) -> str:
    """
    Génère une empreinte hashable à partir des paramètres (à l'exclusion de l'ID).
    """
    return chr(30).join([f"{key}{chr(31)}{params[key]}" for key in sorted(params) if key != 'id'])
