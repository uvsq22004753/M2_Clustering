import hdbscan
import logging
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matchms.importing import load_from_mgf
from MASS_SPECTRA.fast_spectra_distances import read_matrix

## attention : gère très mal le fait que l'on ait des points identiques
## attention : des points considérés comme du bruit dans le cluster -1

def list_smiles(input_file):
    """
    Extrait les SMILES d'un fichier MGF et les stocke dans une liste.

    Arguments
    ----------
    input_file : str
        Chemin du fichier MGF contenant les spectres.

    Retourne
    -------
    list of str
        Liste des SMILES extraits du fichier.
    """
    
    logging.getLogger("matchms").setLevel(logging.ERROR)
    res = list()

    for spectrum in list(load_from_mgf(input_file)):
        res.append(spectrum.metadata.get("smiles"))

    return res


def save_results_hdbscan(data, outputfile):
    """
    Sauvegarde les résultats du clustering HDBSCAN dans un fichier JSON.

    Arguments
    ----------
    data : list of int
        Liste des labels de cluster obtenus avec HDBSCAN.
    outputfile : str
        Chemin du fichier JSON où enregistrer les résultats.

    Retourne
    -------
    None
    """

    clusters_dict = {}

    for indice, cluster in enumerate(data):
        
        if cluster not in clusters_dict:
            clusters_dict[cluster] = []
        
        clusters_dict[cluster].append(indice)

    clusters_dict_str = {str(k): v for k, v in clusters_dict.items()}

    # Enregistrement
    with open(outputfile, "w") as json_file:
        json.dump(clusters_dict_str, json_file, indent=4)


def spectre_hdbscan(inputfile, outputfile, min_cluster_size, min_samples, cluster_selection_epsilon=0.0):
    """
    Applique HDBSCAN sur une matrice de distances et sauvegarde les résultats.

    Arguments
    ----------
    inputfile : str
        Chemin du fichier contenant la matrice de distances.
    outputfile : str
        Chemin du fichier JSON où enregistrer les labels de clustering.
    min_cluster_size : int
        Nombre minimum de points requis pour former un cluster.
    min_samples : int
        Nombre minimum de voisins pour qu'un point soit un "core point".
    cluster_selection_epsilon : float, optional (default=0.0)
        Seuil de distance pour la sélection des clusters (réduit la fragmentation des clusters).

    Retourne
    -------
    None
    """

    data = read_matrix(inputfile)

    clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=cluster_selection_epsilon)
    clusterer.fit(data)
    
    save_results_hdbscan(clusterer.labels_, outputfile)