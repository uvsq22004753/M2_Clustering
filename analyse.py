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

def parametres():
    """
    Analyse les paramètres du fichier ALL_GNPS_cleaned.mgf

    Retourne
    -------
    set
        les noms des différentes metadata dans le fichier.
    """
    file = "./ALL_GNPS_cleaned.mgf"
    all_param = set()

    for spectrum in mgf.read(file, use_index=False):
        all_param.update(spectrum['params'].keys())

    print(all_param)


def statistiques():
    """
    Analyse un fichier MGF et calcule le pourcentage de spectres contenant certains paramètres MS.
    """

    file = './ALL_GNPS_cleaned.mgf'
    total_spectra = 542777  # Nombre total de spectres

    stats = {
        "adduct": 0,
        "collision_energy": 0,
        "all": 0,
        "ms_dissociation": [0, set()],
        "ms_mass_analyzer": [0, set()],
        "ms_manufacturer": [0, set()],
        "ms_ionisation": [0, set()]
    }

    for spectrum in mgf.read(file, use_index=False):
        
        params = spectrum['params']

        # Vérifie si tous les paramètres sont présents simultanément
        if all(k in params for k in ["ms_ionisation", "ms_manufacturer", "ms_dissociation", "ms_mass_analyzer", "collision_energy"]):
            stats["all"] += 1

        # Mise à jour des statistiques
        for key in ["adduct", "collision_energy"]:
            if key in params:
                stats[key] += 1

        for key in ["ms_dissociation", "ms_mass_analyzer", "ms_manufacturer", "ms_ionisation"]:
            if key in params:
                stats[key][0] += 1 
                stats[key][1].add(params[key])  # Valeurs uniques

    for key, value in stats.items():
        if isinstance(value, list):
            count, unique_values = value
            percentage = (count / total_spectra) * 100
            print(f"{key} : {count} spectres ({percentage:.2f}%), valeurs uniques : {unique_values}")
        else:
            percentage = (value / total_spectra) * 100
            print(f"{key} : {value} spectres ({percentage:.2f}%)")



if __name__ == "__main__":
    analyse_adduct()