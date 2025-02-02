import os
import pandas as pd

from pyteomics import mgf

def analyse_adduct():
    """
    Analyse les fichiers MGF dans le dossier './adducts' pour compter :
    - Le nombre total de spectres par fichier.
    - Le nombre de SMILES uniques par fichier.
    - Le nombre de combinaisons uniques (SMILES + Compound Name).
    
    Résultats enregistrés dans 'adducts_info.xlsx'.
    """

    dir = "./adducts"
    nbr_spectra = dict()
    nbr_smiles = dict()
    nbr_unique = dict()

    for file in os.listdir(dir):

        print(f"Processing {file} file.")
        adduct = file[:-4]

        filepath = os.path.join(dir, file)
        filepath = os.path.normpath(filepath)

        unique_smiles = set()
        unique_spectrum = 0
        unique_entries = set()
        
        for spectrum in mgf.read(filepath, use_index=False):
            params = spectrum["params"]
            compound_name = params.get('compound_name')
            smiles = params.get('smiles')

            unique_entries.add((smiles, compound_name))
            unique_smiles.add(smiles)
            unique_spectrum += 1

        # Stocker le nombre de SMILES uniques pour cet adduct
        nbr_smiles[adduct] = len(unique_smiles)
        nbr_spectra[adduct] = unique_spectrum
        nbr_unique[adduct] = len(unique_entries)

    table = pd.DataFrame({
    'Adduct': list(nbr_spectra.keys()),
    'Nbr spectres': list(nbr_spectra.values()),
    'Nbr smiles' : list(nbr_smiles.values()), 
    'Nbr unique' : list(nbr_unique.values())
    })
    
    # Enregistrement dans un fichier Excel
    nom_fichier = "adducts_info.xlsx"
    table.to_excel(nom_fichier, index=False)


if __name__ == "__main__":
    analyse_adduct()