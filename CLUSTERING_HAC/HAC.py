"""
Ce fichier permet de faire du clustering hiérarchique (HAC),
à partir de matrice de distance.
"""

import os
from sklearn.cluster import AgglomerativeClustering
from datetime import datetime
import json
from SMILES.fingerprints import fp_matrix_distance
from CLUSTERING_HAC.HAC_utils import readfile_smiles, save_dendogram
from MASS_SPECTRA.cosinus_distance import all_cosinus_distance_binning, list_all_spectra


def HAC(distance_matrix, NB_CLUSTERS = 150, filename="filename"):
    """
    Effectue un clustering hiérarchique sur des molécules représentées 
    à partir d'une matrice de distance carrée.
    Les résultats du clustering sont sauvegardés dans un fichier JSON.

    Arguments
    ----------
    distance_matrix : np.ndarray
        Matrice carrée de distances entre les molécules

    NB_CLUSTERS : int, optional (default=150)
        Nombre de clusters

    Retourne
    -------
    None
        Sauvegarde les résultats du clustering dans un fichier JSON contenant 
        les identifiants des molécules et leurs clusters respectifs.
    """
    # Clustering hiérarchique avec méthode average
    clustering = AgglomerativeClustering(n_clusters=NB_CLUSTERS, metric='precomputed'
                                         , linkage='average')
    clustering.fit(distance_matrix)

    # Sauvegarde des résultats dans un dictionnaire
    cluster_info = {
        "clusters": [
            {"Molecule_ID": index, "Cluster_ID": int(cluster)}
            for index, cluster in enumerate(clustering.labels_)
        ]
    }

    # Sauvegarde dans un fichier JSON
    json_filename = f"HAC_{filename}_fp_cos.json"
    with open(json_filename, "w") as json_file:
        json.dump(cluster_info, json_file, indent=4)



##################### SMILES FINGERPRINTS #####################
# pathname = "CLUSTERING_HAC/[M-H2O+H]1+.smiles"
# filename = os.path.basename(pathname)

# print(f'Programm Running... Started at: {datetime.now()}')
# # Mise des SMILES dans une liste
# liste_smiles = readfile_smiles(pathname)

# # Calcul de la matrice carré des distances
# # Pour Morgan Fingerprint
# matrix_distance = fp_matrix_distance(liste_smiles)

# # Clustering & Stockage .json du résultat du clustering
# HAC(matrix_distance, NB_CLUSTERS=150, filename=filename)

# # Affichage du dendogramme
# # save_dendogram(matrix_distance, filename=filename)

# print(f'Programm Ending at: {datetime.now()}')

##################### SPECTRA #####################
# print(f'Programm Running... Started at: {datetime.now()}')
# pathname = "CLUSTERING_HAC/[M-H2O+H]1+_Bin1.mgf"
# filename = os.path.basename(pathname)
# spectra = list_all_spectra(pathname)
# distance_matrix = all_cosinus_distance_binning(spectra)
# print(distance_matrix)

# HAC(distance_matrix=distance_matrix, NB_CLUSTERS=150)
# print(f'Programm Ending at: {datetime.now()}')
