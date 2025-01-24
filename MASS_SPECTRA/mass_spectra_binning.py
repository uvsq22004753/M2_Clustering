from matchms.importing import load_from_mgf
from matchms.exporting import save_as_mgf
from matchms.filtering import normalize_intensities
from matchms import Spectrum
import numpy as np

def binning(spec, bin_size, opt= 'somme'):
    """
    Effectue un binning sur un spectre en regroupant les m/z en pas 
    et en calculant les intensités correspondantes selon une méthode choisie.

    Arguments
    ----------
    spec_mz : Spectrum
        Un spectre matchms avec metadata, mz, intensities.
    bin_size : float
        Taille des bins (intervalles) pour regrouper les valeurs de m/z. 
        Chaque bin correspond à une plage de m/z commençant à 20 (inclus).
    opt : str, optional
        Méthode d'agrégation des intensités dans chaque bin. Les options possibles sont :
        - "somme" : Les intensités des m/z dans un même bin sont additionnées (par défaut).
        - "moyenne" : Les intensités des m/z dans un même bin sont moyennées.

    Retourne
    -------
    tuple
        - binned_mz (np.ndarray) : Débuts des plages de bins (valeurs de m/z regroupées).
        - binned_intensities (np.ndarray) : Intensités agrégées selon la méthode choisie.

    Raises
    ------
    ZeroDivisionError
        Si la normalisation des intensités est impossible en raison d'intensités nulles.
    """

    spec_mz = spec.peaks.mz
    spec_intensities = spec.peaks.intensities

    # Calcul des indices des bins à partir de 0
    bin_index = np.floor((spec_mz - 20) / bin_size).astype(int)

    # Regrouper les valeurs par bin
    unique_bins, inverse_indices = np.unique(bin_index, return_inverse=True)

    # Initialiser les intensités des bins
    binned_intensities = np.zeros(len(unique_bins), dtype=np.float32)

    # Ajouter les intensités dans chaque bin
    np.add.at(binned_intensities, inverse_indices, spec_intensities)

    # Calculer les moyennes si demandé
    if opt == 'moyenne':
        bin_counts = np.bincount(inverse_indices)
        binned_intensities /= bin_counts

    # Donner un nom aux bins : débuts d'intervalles m/z
    # binned_mz = 20 + unique_bins * bin_size

    # Normaliser les intensités pour les ramener entre 0 et 1
    max_val = np.max(binned_intensities)
    if max_val > 0:
        binned_intensities /= max_val
    else:
        raise ZeroDivisionError("Les intensités sont toutes nulles, normalisation impossible.")

    return Spectrum(mz=unique_bins.astype(float), intensities=binned_intensities, metadata=spec.metadata)


def centroid_binning(spec, bin_size, opt='somme'):
    """
    Effectue un binning centroïde sur un spectre en regroupant les valeurs m/z 
    autour des centroïdes les plus proches en fonction d'une taille de bin donnée.

    Arguments
    ----------
    spec : Spectrum
        Un spectre au format matchms contenant les attributs peaks.mz (m/z), 
        peaks.intensities (intensités) et metadata (métadonnées).
    bin_size : float
        Taille des intervalles entre les centroïdes. Chaque m/z sera assigné
        au centroïde le plus proche, défini comme un multiple de bin_size.
    opt : str, optional
        Méthode d'agrégation des intensités dans chaque bin. Les options possibles sont :
        - "somme" : Additionne les intensités des m/z associés à un même bin (par défaut).
        - "moyenne" : Moyenne les intensités des m/z associés à un même bin.

    Retourne
    -------
    Spectrum
        spectre modifié
    """
    
    spec_mz = spec.peaks.mz
    spec_intensities = spec.peaks.intensities
    
    # Initialiser des dictionnaires pour regrouper les bins
    bin_dict = {}
    count_dict = {}

    for mz, intensity in zip(spec_mz, spec_intensities):
        # Trouver le centroïde le plus proche
        centroid = round(round(mz / bin_size)*bin_size, count_decimal(bin_size))

        if centroid not in bin_dict:
            bin_dict[centroid] = 0
            count_dict[centroid] = 0

        bin_dict[centroid] += intensity
        count_dict[centroid] += 1

    # Convertir les bins en tableaux numpy
    binned_mz = np.array(list(bin_dict.keys()))
    binned_intensities = np.array(list(bin_dict.values()), dtype=np.float32)

    # Moyenne des intensités si demandé
    if opt == 'moyenne':
        counts = np.array([count_dict[mz] for mz in binned_mz])
        binned_intensities /= counts

    res = Spectrum(mz=binned_mz.astype(float), intensities=binned_intensities, metadata=spec.metadata)

    return normalize_intensities(res)


def count_decimal(number):
    """
    Compte le nombre de chiffres après la virgule pour un nombre donné.

    Arguments
    ----------
    number : float
        Le nombre pour lequel on veut compter les chiffres après la virgule.

    Retourne
    -------
    int
        Le nombre de chiffres après la virgule.
    """
    # Convertir en chaîne et supprimer les zéros inutiles
    str_number = str(number).rstrip('0')
    if '.' in str_number:
        return len(str_number.split('.')[1])
    return 0


def padding(spec1_mz, spec1_intensities, spec2_mz ,spec2_intensities):
    """
    Effectue un padding (remplissage) pour deux spectres afin de les aligner sur une plage commune de m/z.

    Cette fonction combine les valeurs m/z des deux spectres et ajuste leurs vecteurs d'intensité en ajoutant des zéros
    pour les m/z qui n'existent pas dans l'un des spectres.

    Arguments
    ----------
    spec1_mz : np.ndarray
        Tableau des valeurs m/z du premier spectre.
    spec1_intensities : np.ndarray
        Tableau des intensités correspondant aux valeurs m/z du premier spectre.
    spec2_mz : np.ndarray
        Tableau des valeurs m/z du deuxième spectre.
    spec2_intensities : np.ndarray
        Tableau des intensités correspondant aux valeurs m/z du deuxième spectre.

    Retourne
    -------
    tuple
        - padded_intensities1 (np.ndarray): Intensités du premier spectre, alignées sur la plage commune de m/z.
        - padded_intensities2 (np.ndarray): Intensités du deuxième spectre, alignées sur la même plage de m/z.

    """

    # Plage commune avec fusion des m/z des deux spectres en éliminant les doublons et en triant les valeurs.
    mz_array = np.union1d(spec1_mz, spec2_mz)

    # Initialiser les vecteurs d'intensité avec des zéros
    padded_intensities1 = np.zeros(len(mz_array), dtype=np.float32)
    padded_intensities2 = np.zeros(len(mz_array), dtype=np.float32)

    # Remplissage due vecteur d'intensités pour le premier spectre
    for mz, intensity in zip(spec1_mz, spec1_intensities):
        index = np.where(mz_array == mz)[0][0]
        padded_intensities1[index] = intensity

    # Remplissage du vecteur d'intensités pour le deuxième spectre
    for mz, intensity in zip(spec2_mz, spec2_intensities):
        index = np.where(mz_array == mz)[0][0]
        padded_intensities2[index] = intensity

    # Retourner les intensités alignées pour les deux spectres
    return padded_intensities1, padded_intensities2


def apply_binning(file_init, file_final, bin, opt='somme', type="centre"):
    """
    Applique un binning (regroupement) sur tous les spectres d'un fichier MGF d'entrée
    et enregistre les spectres modifiés dans un fichier MGF de sortie.

    Paramètres :
    ----------
    file_init : str
        Chemin vers le fichier MGF d'entrée contenant les spectres bruts.
    file_final : str
        Chemin vers le fichier MGF de sortie où seront enregistrés les spectres modifiés.
        Si le fichier existe déjà, son contenu sera effacé avant d'écrire les nouveaux spectres.
    bin : float
        Taille des bins pour le regroupement des valeurs m/z.
        Chaque bin regroupe les m/z dans des intervalles de largeur spécifiée par `bin`.
    opt : str, optional
        Méthode d'agrégation des intensités dans chaque bin (par défaut = 'somme').
        - "somme" : Les intensités dans chaque bin sont additionnées.
        - "moyenne" : Les intensités dans chaque bin sont moyennées.

    Retourne :
    ---------
    None
    """

    spectra = list(load_from_mgf(file_init))
    if type == 'centre':
        spectr_binned = [centroid_binning(spec, bin, opt) for spec in spectra]
    elif type == 'pas':
        spectr_binned = [binning(spec, bin, opt) for spec in spectra]

    open(file_final, "w").close()

    save_as_mgf(spectr_binned, file_final)
