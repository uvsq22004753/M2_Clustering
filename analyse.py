import os
import pandas as pd

from pyteomics import mgf

def analyse_adduct():

    dir = "./adducts"
    nbr_fichier = 0
    nbr_spectra = dict()
    for file in os.listdir(dir):

        print(f"Processing {file} file.")
        nbr_fichier += 1
        adduct = file[:-4]

        filepath = os.path.join(dir, file)
        filepath = os.path.normpath(filepath)

        nbr_spectra[adduct] = len(list(mgf.read(filepath, use_index=False)))

    table = pd.DataFrame({
    'Adduct': list(nbr_spectra.keys()),
    'Nbr spectres': list(nbr_spectra.values())
    })
    
    # Enregistrement dans un fichier Excel
    nom_fichier = "adducts_info.xlsx"
    table.to_excel(nom_fichier, index=False)

analyse_adduct()