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
    return list(load_from_mgf(file))
