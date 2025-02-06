import numpy as np
import os, shutil, time
from matchms.exporting import save_as_mgf
from utils.file_utils import load_mgf_file
from matchms import Spectrum
import logging
from utils.file_utils import new_dir
import config

def binning(spec, bin_size, opt='somme'):
    spec_mz = spec.peaks.mz
    spec_intensities = spec.peaks.intensities
    bin_index = np.floor((spec_mz - config.MZ_FROM) / bin_size).astype(int)
    unique_bins, inverse_indices = np.unique(bin_index, return_inverse=True)
    binned_intensities = np.zeros(len(unique_bins), dtype=np.float32)
    np.add.at(binned_intensities, inverse_indices, spec_intensities)
    if opt == 'moyenne':
        bin_counts = np.bincount(inverse_indices)
        binned_intensities /= bin_counts
    max_val = np.max(binned_intensities)
    if max_val > 0:
        binned_intensities /= max_val
    else:
        raise ZeroDivisionError("Intensities are all zero; cannot normalize.")
    return Spectrum(mz=unique_bins.astype(float), intensities=binned_intensities, metadata=spec.metadata)

def all_mass_spectra_binning(inputdir, outputdir, bin_size=1, opt='somme'):
    """
    Parcourt le répertoire inputdir, applique le binning à chaque fichier MGF et sauvegarde dans outputdir.
    """
    new_dir(outputdir)
    for file in os.listdir(inputdir):
        if file.lower().endswith(".mgf"):
            file_path = os.path.join(inputdir, file)
            print(f"Processing {file}...")
            deb = time.time()
            spectra = list(load_mgf_file(file_path))
            binned_spectra = [binning(spec, bin_size, opt) for spec in spectra]
            output_file = file[:-4] + f"_Bin{str(bin_size).replace('.', '')}.mgf"
            output_file_path = os.path.join(outputdir, output_file)
            save_as_mgf(binned_spectra, output_file_path)
            print(f"Execution in {time.time()-deb:.2f} s.")
