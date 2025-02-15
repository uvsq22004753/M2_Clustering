import os
import shutil
import logging
from matchms.importing import load_from_mgf

def new_dir(directory: str):
    """
    Crée un répertoire en supprimant celui existant si l'utilisateur le confirme.
    Utilise os.makedirs pour créer tous les dossiers parents nécessaires.
    """
    if os.path.exists(directory):
        response = input(f"[WARNING] Remove '{directory}' directory? [y/N] ").strip().lower()
        if response == 'y':
            shutil.rmtree(directory)
        else:
            logging.error("Aborting...")
            exit(0)
    # Utilise os.makedirs pour créer tous les parents (exist_ok=False car on vient de supprimer)
    os.makedirs(directory, exist_ok=True)


def load_mgf_file(file: str) -> list:
    """
    Charge le fichier MGF en utilisant matchms et retourne une liste de spectra.

    Paramètres:
      - file (str): Chemin du fichier MGF.
      
    Retourne:
      - list: La liste des spectra chargés.
    """
    logging.getLogger("matchms").setLevel(logging.ERROR)
    return list(load_from_mgf(file))


def read_smiles_file(filename: str) -> list:
    """
    Lit un fichier texte contenant des SMILES (un SMILES par ligne) et retourne une liste de SMILES.
    
    Paramètres:
      - filename (str): Chemin du fichier texte.
    
    Retourne:
      - list: Une liste de SMILES.
    """
    smiles = []
    with open(filename, "r") as file:
        for line in file:
            s = line.strip()
            if s:
                smiles.append(s)
    return smiles