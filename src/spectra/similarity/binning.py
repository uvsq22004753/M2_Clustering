import os
import time
import numpy as np
from matchms.exporting import save_as_mgf
from utils.file_utils import load_mgf_file, new_dir
from matchms import Spectrum
import logging
import config

def binning(spec, bin_size, opt='somme'):
    """
    Applique le binning sur un spectre avec normalisation.

    Les intensités des bins sont agrégées (addition ou moyenne) puis normalisées par la valeur maximale.
    
    Arguments:
      - spec: Spectrum (doit posséder spec.peaks.mz et spec.peaks.intensities).
      - bin_size: float, largeur du bin.
      - opt: str, méthode d'agrégation; 'somme' (par défaut) ou 'moyenne'.
    
    Retourne:
      - Spectrum: objet Spectrum avec les valeurs de m/z (les bins uniques) et les intensités normalisées.
    """
    spec_mz = spec.peaks.mz
    spec_intensities = spec.peaks.intensities
    # Calcul du bin index à partir de config.MZ_FROM pour rester cohérent
    bin_index = (spec_mz - config.MZ_FROM) // bin_size
    bin_index = bin_index.astype(int)
    unique_bins, inverse_indices = np.unique(bin_index, return_inverse=True)
    binned_intensities = np.zeros(len(unique_bins), dtype=np.float32)
    np.add.at(binned_intensities, inverse_indices, spec_intensities)
    
    if opt == 'moyenne':
        bin_counts = np.bincount(inverse_indices)
        binned_intensities /= bin_counts

    # Normalisation par la valeur maximale
    max_val = np.max(binned_intensities)
    if max_val > 0:
        binned_intensities /= max_val
    else:
        raise ZeroDivisionError("Intensities are all zero; cannot normalize.")
    
    return Spectrum(mz=unique_bins.astype(float), intensities=binned_intensities, metadata=spec.metadata)

def fixed_binning_vector(spec, bin_size, mz_min=20, mz_max=2000):
    """
    Calcule un vecteur de caractéristiques pour un spectre à l'aide d'un binning fixe
    sans normalisation (pour être ensuite normalisé dans le pipeline de clustering).
    
    Arguments :
      - spec      : Objet Spectrum (issu de matchms) avec spec.peaks.mz et spec.peaks.intensities.
      - bin_size  : Largeur d'un bin (float).
      - mz_min    : Valeur minimale de m/z (défaut=20).
      - mz_max    : Valeur maximale de m/z (défaut=2000).
    
    Retourne :
      - Un vecteur numpy (1D) de dimension (n_bins,).
    """
    bins = np.arange(mz_min, mz_max + bin_size, bin_size)
    feature, _ = np.histogram(spec.peaks.mz, bins=bins, weights=spec.peaks.intensities)
    return feature.astype(float)


def bin_file(input_file: str, output_dir: str, bin_size: float = 1, opt: str = 'somme') -> str:
    """
    Applique le binning (avec normalisation) sur un fichier MGF unique et sauvegarde le résultat dans output_dir.

    Cette fonction utilise la méthode 'binning' (qui intègre la normalisation).

    Arguments:
      - input_file (str): chemin complet du fichier MGF à traiter.
      - output_dir (str): dossier où sauvegarder le fichier binned.
      - bin_size (float): taille du bin (par défaut 1).
      - opt (str): méthode d'agrégation ('somme' ou 'moyenne').

    Retourne:
      - str: chemin complet du fichier binned généré.
    """
    new_dir(output_dir)
    file = os.path.basename(input_file)
    print(f"Processing {file} ...")
    deb = time.time()
    spectra = list(load_mgf_file(input_file))
    binned_spectra = [binning(spec, bin_size, opt) for spec in spectra]
    base_name = file[:-4]  # on retire l'extension .mgf
    output_file = f"{base_name}_Bin{bin_size}.mgf"
    output_file_path = os.path.join(output_dir, output_file)
    save_as_mgf(binned_spectra, output_file_path)
    print(f"Binning execution in {time.time()-deb:.2f} s.")
    return output_file_path
