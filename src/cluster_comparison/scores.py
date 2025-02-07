import json
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def open_json(file_path: str) -> dict:
    """
    Ouvre un fichier JSON contenant les résultats de clustering et retourne un dictionnaire 
    où chaque clé est le label de cluster et la valeur est la liste des identifiants de molécules.
    
    On suppose que le fichier JSON contient une clé "results" qui est une liste d'objets
    avec les clés "id" et "cluster".
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    results = data.get("results", [])
    clusters = {}
    for item in results:
        cid = item["cluster"]
        mid = item["id"]
        if cid not in clusters:
            clusters[cid] = []
        clusters[cid].append(mid)
    return clusters

def transform_dict_cluster_to_list(clusters_dict: dict) -> list:
    """
    Transforme un dictionnaire de clusters en une liste telle que
    l'indice de la liste correspond à l'identifiant de la molécule 
    et la valeur à l'identifiant du cluster auquel elle appartient.
    
    Si une molécule n'appartient à aucun cluster, la valeur par défaut est -1.
    """
    # Conversion de tous les identifiants en int pour être sûr
    num_molecules = max(max(int(mol_id) for mol_id in cluster) for cluster in clusters_dict.values()) + 1
    cluster_list = [-1] * num_molecules  # valeur par défaut: -1 (non affecté)
    
    # Remplit la liste selon le mapping des clusters.
    for cluster_id, molecule_ids in clusters_dict.items():
        for molecule_id in molecule_ids:
            cluster_list[int(molecule_id)] = cluster_id
    return cluster_list


def ARI(cluster_list1: list, cluster_list2: list) -> float:
    """
    Calcule l'Indice de Rand Ajusté (ARI) entre deux partitions.
    
    Arguments:
      - cluster_list1: list des labels de clusters pour chaque molécule.
      - cluster_list2: list des labels de clusters pour chaque molécule.
    
    Retourne:
      - float: ARI.
    """
    return adjusted_rand_score(cluster_list1, cluster_list2)

def NMI(cluster_list1: list, cluster_list2: list) -> float:
    """
    Calcule la Normalized Mutual Information (NMI) entre deux partitions.
    
    Arguments:
      - cluster_list1: list des labels de clusters pour chaque molécule.
      - cluster_list2: list des labels de clusters pour chaque molécule.
    
    Retourne:
      - float: NMI.
    """
    return normalized_mutual_info_score(cluster_list1, cluster_list2)

def compare_clusterings(file1: str, file2: str):
    """
    Lit deux fichiers JSON de clustering, transforme leur contenu en listes de labels,
    et affiche les scores ARI et NMI.
    """
    clusters_dict1 = open_json(file1)
    clusters_dict2 = open_json(file2)
    
    labels1 = transform_dict_cluster_to_list(clusters_dict1)
    labels2 = transform_dict_cluster_to_list(clusters_dict2)
    
    ari_score = ARI(labels1, labels2)
    nmi_score = NMI(labels1, labels2)
    
    print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <cluster_file1.json> <cluster_file2.json>")
        sys.exit(1)
    compare_clusterings(sys.argv[1], sys.argv[2])
