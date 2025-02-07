import hdbscan
import logging
import json
import numpy as np
import hdbscan
import logging
import json
import numpy as np

from scipy.spatial.distance import pdist, squareform
from matchms.importing import load_from_mgf
from MASS_SPECTRA.fast_spectra_distances import read_matrix
from SMILES.python.fingerprints import morgan_fingerprint

## pour les smiles faire une étape pour enlever tous les doublons

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

def read_smiles(file):
    """
    Lit un fichier contenant des SMILES, élimine les doublons et construit une correspondance entre SMILES et leurs indices initiaux.

    Arguments
    ----------
    file : str
        Chemin du fichier contenant les données SMILES.

    Retourne
    -------
    smiles : list of str
        Liste des SMILES uniques trouvés dans le fichier.
    correspondance : dict
        Dictionnaire associant chaque SMILES unique à la liste de ses indices dans le fichier original.
    taille : int
        Nombre total de SMILES dans le fichier.
    """

    smiles = list_smiles(file)
    taille = len(smiles)
    correspondance = {}

    # Parcours des données et assignation aux groupes uniques
    for i, smile in enumerate(smiles):
        
        if smile not in correspondance:
            correspondance[smile] = [] 
        
        correspondance[smile].append(i)

    smiles = list(correspondance.keys())

    return smiles, correspondance, taille


def add_copies_result(dict, data, taille):
    """
    Associe chaque élément d'un ensemble de données au cluster qui lui correspond.

    Arguments
    ----------
    dict : dict
        Dictionnaire associant chaque élément unique à ses indices dans l'ensemble de données.
    data : list
        Liste des labels de clusters obtenus après clustering.
    taille : int
        Nombre total d'éléments dans l'ensemble de données.

    Retourne
    -------
    list of int
        Liste des labels de clusters attribués à chaque élément initial.
    """

    res = [0 for _ in range(taille)]
    ind = 0

    for _, copies in dict.items():
        cluster = data[ind]
        for elem in copies:
            res[elem] = int(cluster)

        ind +=1

    return res


def save_results_hdbscan(data, nbr_cluster, outputfile):
    """
    Sauvegarde les résultats du clustering HDBSCAN dans un fichier JSON.
    Les points considérés comme du bruit sont assignés à des clusters uniques.

    Arguments
    ----------
    data : list of int
        Liste des labels de clusters obtenus après clustering.
    nbr_cluster : int
        Nombre initial de clusters détectés.
    outputfile : str
        Chemin du fichier JSON où enregistrer les résultats.

    Retourne
    -------
    None
    """

    clusters_dict = {}
    nbr_cluster = nbr_cluster

    for indice, cluster in enumerate(data):
        if int(cluster) == -1:
            nbr_cluster += 1
            clusters_dict[nbr_cluster] = [indice]

        else :
            if cluster not in clusters_dict:
                clusters_dict[cluster] = []
        
            clusters_dict[cluster].append(indice)

    clusters_dict_str = {str(k): v for k, v in clusters_dict.items()}

    # Enregistrement
    with open(outputfile, "w") as json_file:
        json.dump(clusters_dict_str, json_file, indent=4)


def apply_hdbscan(matrix, min_cluster_size, min_samples):
    """
    Applique l'algorithme HDBSCAN sur une matrice de distance pour effectuer du clustering.

    Arguments
    ----------
    matrix : array-like
        Matrice de distances utilisée pour le clustering.
    min_cluster_size : int
        Taille minimale d'un cluster.
    min_samples : int
        Nombre minimum d'échantillons pour considérer un cluster dense.

    Retourne
    -------
    tuple
        - Liste des labels de clusters attribués à chaque élément.
        - Nombre de clusters détectés.
    """

    clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=min_cluster_size, min_samples=min_samples)
    clusterer.fit(matrix)
    
    #compte le nombre d'éléments considéré comme du bruit
    #print(np.count_nonzero(clusterer.labels_ == -1))
    
    return clusterer.labels_, clusterer.labels_.max()


def _hdbscan(inputfile, outputfile, option, min_cluster, min_densite):
    """
    Exécute le pipeline de clustering HDBSCAN sur des données moléculaires en fonction de l'option choisie.

    Arguments
    ----------
    inputfile : str
        Chemin du fichier d'entrée contenant les données à analyser.
    outputfile : str
        Chemin du fichier où enregistrer les résultats.
    option : str
        Mode de clustering : "smiles" pour les structures chimiques ou "spectre" pour les spectres de masse.
    min_cluster : int
        Taille minimale des clusters.
    min_densite : int
        Densité minimale pour la formation de clusters.

    Retourne
    -------
    None
    """
     
    if option == "smiles":
        # on enlève les doublons pour faire le clustering sur SMILES
        smiles, smiles_doublons_correspondances, taille = read_smiles(inputfile)
        
        # on applique la distance cosinus sur les fingerprints
        fingerprints = [morgan_fingerprint(smile) for smile in smiles]

        # on forme une matrice carrée symétrique
        distance_smiles = pdist(fingerprints, metric="cosine")
        matrix = squareform(distance_smiles, "tomatrix")

        clustering, nblabel = apply_hdbscan(matrix, min_cluster, min_densite)
        res = add_copies_result(smiles_doublons_correspondances, clustering, taille)
        save_results_hdbscan(res, nblabel, outputfile)

    elif option == "spectre":

        matrix = read_matrix(inputfile)
        clustering, nblabel = apply_hdbscan(matrix, min_cluster, min_densite)
        save_results_hdbscan(clustering, nblabel, outputfile)
