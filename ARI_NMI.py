from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from cmp_cluster import *
import os

SMILE_PATH = "clusters/smiles"
SPECTRE_PATH = "clusters/spectres"


def get_all_files_in_dir(directory):
    files = []
    for entry in os.listdir(directory):
        entry_path = os.path.join(directory, entry)
        if os.path.isfile(entry_path):
            files.append(entry_path)
    return files


def transform_dict_cluster_to_list(clusters_dict):
    """
    Transforme un dictionnaire de clusters 
    (issu de la fonction open_json de comp_cluster) 
    en une liste où chaque indice représente une molécule 
    et la valeur de cet indice représente l'identifiant du cluster 
    auquel la molécule appartient.

    Arguments:
        clusters_dict (dict): Un dictionnaire où les clés sont des 
                            identifiants de clusters et les valeurs 
                            sont des listes des indices de molécules 
                            appartenant à chaque cluster.

    Retourne:
        list: Une liste dans laquelle chaque élément représente 
              l'identifiant du cluster de la molécule
              correspondant à cet indice.
    """
    # Initialisation de la liste vide pour les clusters
    num_molecules = max(max(cluster) for cluster in clusters_dict.values()) + 1
    cluster_list = [-1] * num_molecules  # Valeur cluster par défaut: -1 (= non affecté)

    # Remplissage de la liste selon dict
    for cluster_id, molecules in clusters_dict.items():
        for molecule_id in molecules:
            cluster_list[molecule_id] = cluster_id

    return(cluster_list)


def ARI(smiles_clusters, spectra_clusters):
    """
    Calcule l'Indice de Rand Ajusté (ARI) entre deux ensemble de clusters. 

    Arguments:
        smiles_clusters (list): Liste des identifiants de clusters pour les molécules
        spectra_clusters (list): Liste des identifiants de clusters pour les mêmes molécules

    Retourne:
        float: La valeur de l'Indice de Rand Ajusté entre les deux clusters
    """
    return adjusted_rand_score(smiles_clusters, spectra_clusters)

def NMI(smiles_clusters, spectra_clusters):
    """
    Calcule l'Information Mutuelle Normalisée (NMI) entre deux ensemble de clusters 

    Arguments:
        smiles_clusters (list): Liste des identifiants de clusters pour les molécules
        spectra_clusters (list): Liste des identifiants de clusters pour les mêmes molécules

    Returns:
        float: La valeur de l'Information Mutuelle Normalisée entre les deux ensemble de clusters.
    """
    return normalized_mutual_info_score(smiles_clusters, spectra_clusters)


def main():
    FILE_SMILES = sorted(get_all_files_in_dir(SMILE_PATH))
    FILE_SPECTRE = sorted(get_all_files_in_dir(SPECTRE_PATH))

    for i in range(min(len(FILE_SMILES), len(FILE_SPECTRE))):
        clusters1 = open_json(FILE_SMILES[i])
        clusters2 = open_json(FILE_SPECTRE[i])

        clusters1 = transform_dict_cluster_to_list(clusters1)
        clusters2 = transform_dict_cluster_to_list(clusters2)

        print(f"Results : {FILE_SMILES[i][16:-11]}")
        

        ari = ARI(clusters1, clusters2)
        nmi = NMI(clusters1, clusters2)
        print(f"{round(ari, 3)} | {round(nmi, 3)}")
        

########################################################################

if __name__ == '__main__':
    main()