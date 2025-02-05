from matchms.importing import load_from_mgf
from matchms.exporting import save_as_mgf
from matchms import Spectrum

import os
import time
import numpy as np
import shutil
import logging

# on affiche pas les erreurs de matchms
logging.getLogger("matchms").setLevel(logging.ERROR)


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
    Spectrum
        nouveau spectre matchms

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

    # Normaliser les intensités pour les ramener entre 0 et 1
    max_val = np.max(binned_intensities)
    if max_val > 0:
        binned_intensities /= max_val
    else:
        raise ZeroDivisionError("Les intensités sont toutes nulles, normalisation impossible.")

    return Spectrum(mz=unique_bins.astype(float), intensities=binned_intensities, metadata=spec.metadata)


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


def new_dir(directory):
    """
    Crée un répertoire ou supprime et recrée un répertoire existant.

    Cette fonction vérifie si le répertoire spécifié existe. Si c'est le cas, 
    elle demande à l'utilisateur s'il souhaite le supprimer. Si l'utilisateur 
    accepte, le répertoire est supprimé et recréé. Sinon, le programme s'arrête.
    Si le répertoire n'existe pas, il est simplement créé.

    Arguments
    ----------
    directory : str
        Chemin du répertoire à créer ou recréer.

    Retourne
    -------
    None
    """
    if os.path.exists(directory):
        if input(f"[WARNING] Remove '{directory}' directory ? [y/N] ").lower()=='y':
            shutil.rmtree(directory)
        else:
            print("Aborting...")
            exit(0)
    os.mkdir(directory)


def mass_spectra_binning(input, output, bin_size=1, opt='somme'):
    """
    Regroupe les masses des spectres (binning) dans des fichiers MGF et sauvegarde les résultats.

    Arguments
    ----------
    input : str
        Chemin du fichier MGF à traiter
    output : str
        Chemin du fichier MGF dans lequel on stockera les spectres binned
    bin_size : float, optionnel
        Taille des bins (intervalle de regroupement des masses) (par défaut 1).
    opt : str, optionnel
        Méthode de regroupement utilisée, par exemple 'somme' pour sommer les intensités (par défaut 'somme').

    Retourne
    -------
    None
    """

    deb = time.time()

    spectra = list(load_from_mgf(input))
    binned_spectra = [binning(spec, bin_size, opt) for spec in spectra]

    save_as_mgf(binned_spectra, output)
    print(f"Execution in {time.time()-deb} s.")


def all_mass_spectra_binning(inputdir="./adducts", bin_size=1, opt='somme'):
    """
    Regroupe les masses des spectres (binning) dans des fichiers MGF et sauvegarde les résultats.

    Cette fonction parcourt les fichiers MGF dans un répertoire d'entrée, applique un binning
    sur les spectres selon une taille spécifiée, puis sauvegarde les spectres regroupés dans un
    nouveau répertoire. Chaque fichier traité génère un fichier MGF correspondant avec les données 
    regroupées.

    Arguments
    ----------
    inputdir : str, optionnel
        Chemin du répertoire contenant les fichiers MGF à traiter (par défaut "./adducts").
    bin_size : float, optionnel
        Taille des bins (intervalle de regroupement des masses) (par défaut 1).
    opt : str, optionnel
        Méthode de regroupement utilisée, par exemple 'somme' pour sommer les intensités (par défaut 'somme').

    Retourne
    -------
    None
    """

    outputdir = inputdir + "Bin" + str(bin_size).replace(".", "")
    new_dir(outputdir)

    for file in os.listdir(inputdir):

        print(f"Processing {file} file...")
        
        deb = time.time()
        input_file_path = os.path.join(inputdir, file)
        input_file_path = os.path.normpath(input_file_path)
        
        spectra = list(load_from_mgf(input_file_path))
        binned_spectra = [binning(spec, bin_size, opt) for spec in spectra]

        output_file = file[:-4] + "_Bin" + str(bin_size).replace('.', "") + ".mgf"
        output_file_path = os.path.join(outputdir, output_file)
        output_file_path = os.path.normpath(output_file_path)

        save_as_mgf(binned_spectra, output_file_path)
        print(f"Execution in {time.time()-deb} s.")

if __name__ == '__main__':
    
    mass_spectra_binning()