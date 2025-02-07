import hdbscan
import numpy as np

def apply_hdbscan(matrix, min_cluster_size, min_samples):
    """
    Applique l'algorithme HDBSCAN sur une matrice de distance pour effectuer du clustering.

    Arguments:
      - matrix : array-like
          Matrice de distances utilisée pour le clustering.
      - min_cluster_size : int
          Taille minimale d'un cluster.
      - min_samples : int
          Nombre minimum d'échantillons pour considérer un cluster dense.

    Retourne:
      - tuple:
          labels : liste des labels de clusters attribués à chaque élément.
          max_label : nombre maximum de cluster (le label le plus élevé).
    """
    clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=min_cluster_size, min_samples=min_samples)
    clusterer.fit(matrix)
    return clusterer.labels_, int(np.max(clusterer.labels_))
