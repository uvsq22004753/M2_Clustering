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

def fixed_binning(spec, bin_size, mz_min=config.MZ_FROM, mz_max=config.MZ_TO):
    """
    Calcule un vecteur de caractéristiques pour un spectre à l'aide d'un fixed binning
    sans normalisation.
    
    Chaque bin est défini sur l'axe m/z entre mz_min et mz_max avec une largeur fixée par bin_size.
    Pour chaque bin, la valeur correspond à la somme des intensités des pics dont le m/z tombe dans ce bin.
    
    Arguments:
      - spec: Spectrum (doit posséder spec.peaks.mz et spec.peaks.intensities).
      - bin_size: float, largeur d'un bin.
      - mz_min: float, valeur minimale de m/z (défaut=20).
      - mz_max: float, valeur maximale de m/z (défaut=2000).
    
    Retourne:
      - Spectrum: un objet Spectrum dont les m/z correspondent aux bornes inférieures de chaque bin,
                  et les intensités sont la somme des intensités dans chaque bin, sans normalisation.
    """
    bins = np.arange(mz_min, mz_max + bin_size, bin_size)
    feature, _ = np.histogram(spec.peaks.mz, bins=bins, weights=spec.peaks.intensities)
    return Spectrum(mz=bins[:-1].astype(float), intensities=feature, metadata=spec.metadata)

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
