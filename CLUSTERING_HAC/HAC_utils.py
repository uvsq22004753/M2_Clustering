"""
Ce fichier regroupe les fonctions utiles aux fonctionnements
du clustering HAC.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

def readfile_smiles(filename):
    """
    Lit un fichier contenant des SMILES et renvoie une liste des SMILES.

    Arguments
    ----------
    filename : str
        Le nom du fichier contenant UN SMILES par ligne.

    Retourne
    -------
    list
        Une liste de string représentant les SMILES extraits du fichier. 
        Exemple: [SMILES1, SMILES2, SMILES3, ...]
    """

    liste_smiles = []

    # Lecture du fichier
    with open(filename, "r") as file:
        lignes = file.readlines()

        for ligne in lignes:
            ligne = ligne.strip()
            liste_smiles.append(ligne)
    
    return liste_smiles

def elbow_method(distance_matrix, min_clusters, max_clusters, pas=10, filename="filename"):
    """
    Applique la méthode d'Elbow (du coude) pour déterminer le nombre optimal
    de clusters et enregistre la courbe.

    Arguments
    ----------
    liste_smiles : list
        Liste des SMILES pour le clustering
    min_clusters : int
        Nombre minimum de clusters à tester
    max_clusters : int
        Nombre maximum de clusters à tester
    pas : int, optional
        Pas entre chaque test de nombre de clusters (par défaut 10).

    Retourne
    -------
    None
        Sauvergarde de la courbe d'Elbow.
    """
    # Chemin de sauvegarde de la courbe
    save_path = (f"CLUSTERING_HAC/Elbow.jpg")

    # Liste des distances intra-cluster
    intra_distances = []
    
    # Calcul somme des distances intra-cluster pour différents nombres de clusters
    for nb_clusters in range(min_clusters, max_clusters + 1, pas):
        clustering = AgglomerativeClustering(n_clusters=nb_clusters, metric='precomputed'
                                             , linkage='average')
        clustering.fit(distance_matrix)
        
        # Calcul somme des distances intra-cluster
        total_distance = 0
        for cluster_id in range(nb_clusters):
            cluster_points = distance_matrix[clustering.labels_ == cluster_id]
            
            # Calcul distances entre tous les points du cluster
            distance_smiles = pdist(cluster_points, metric="cosine")
            total_distance += np.sum(distance_smiles)  # Somme distances intra-cluster
        
        intra_distances.append(total_distance)
    
    # Tracage et sauvegarde courbe Elbow
    plt.figure(figsize=(8, 6))
    plt.plot(range(min_clusters, max_clusters + 1, pas), intra_distances
             , marker='o', linestyle='-')
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Somme des distances intra-cluster')
    plt.title(f'Courbe méthode d\'Elbow Method pour le fichier {filename}')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Courbe méthode d'Elbow sauvegardée sous '{save_path}'")


def save_dendogram(distance_matrix, filename="filename"):
    """
    Savugarde le dendogramme du clustering HAC.
    ! (Au dessus de 100 molécules, peu lisible) !

    Arguments
    ----------
    distance_matrix : np.ndarray
        Matrice carrée de distances entre les molécules

    Retourne
    -------
    None
        Enregistre le dendrogramme à l'écran.
    """

    plt.figure(figsize=(10, 7))
    sch.dendrogram(sch.linkage(squareform(distance_matrix), method='average'))
    plt.title(f"Dendrogramme HAC du fichier {filename}")
    plt.xlabel("ID molécules")
    plt.ylabel("Distance cosinus")

    # Sauvegarde du dendrogramme
    saving_filename = (f"CLUSTERING_HAC/Dendogramme_{filename}.jpg")
    plt.savefig(saving_filename, dpi=300, bbox_inches='tight')  
    print(f"Dendrogramme sauvegardé sous : {filename}")

