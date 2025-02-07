from sklearn.cluster import AgglomerativeClustering

def run_hac(distance_matrix, n_clusters):
    """
    Exécute le clustering hiérarchique agglomératif (HAC) sur une matrice de distances pré-calculée.
    
    Paramètres:
      - distance_matrix (numpy.ndarray): matrice de distances pré-calculée (symétrique).
      - n_clusters (int): nombre de clusters à former.
      
    Retourne:
      - labels (numpy.ndarray): tableau des labels de clusters.
    """
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    clustering.fit(distance_matrix)
    return clustering.labels_
