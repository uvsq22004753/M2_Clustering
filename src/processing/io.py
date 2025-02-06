import os
import logging
from utils.file_utils import new_dir

def write_mgf_file(spectra: list, output_file: str, mgf_module):
    """
    Écrit la liste des spectra dans un fichier MGF.
    """
    with mgf_module.write(spectra, output_file) as writer:
        pass

def write_smiles_file(spectra: list, output_file: str):
    """
    Écrit un fichier SMILES contenant uniquement le SMILES par ligne.
    """
    with open(output_file, "w") as f:
        for spectrum in spectra:
            f.write(f"{spectrum['params']['smiles']}\n")
