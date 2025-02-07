import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging
import os

logger = logging.getLogger(__name__)

def read_cluster_json(file_path: str) -> dict:
    """
    Lit un fichier JSON de clustering et retourne un dictionnaire où
    chaque clé est l'ID du cluster et la valeur est la liste des IDs des molécules.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    results = data.get("results", [])
    clusters = {}
    for item in results:
        cid = item["cluster"]
        mid = item["id"]
        if cid not in clusters:
            clusters[cid] = []
        clusters[cid].append(int(mid))
    return clusters

def total_molecules(clusters: dict) -> int:
    """Retourne le nombre total de molécules dans le dictionnaire de clusters."""
    return sum(len(ids) for ids in clusters.values())

def compare_clusters(clusters1: dict, clusters2: dict):
    """
    À partir de deux dictionnaires de clusters, calcule des coordonnées fictives pour
    représenter les centroids et les points de chaque molécule.
    
    Les clusters du premier fichier (clusters1) définiront les centroids,
    et ceux du second (clusters2) détermineront la couleur assignée à chaque molécule.
    
    Retourne :
      - centroids : np.ndarray de forme (n_clusters, 2)
      - points : np.ndarray de forme (n_total, 2)
      - colors : np.ndarray de forme (n_total,)
    """
    total = total_molecules(clusters1)
    n_clusters = len(clusters1)
    n_side = int(np.ceil(np.sqrt(n_clusters)))
    s1 = 10  # échelle pour les centroids
    centroids = []
    points = [None] * total
    colors = [None] * total

    # Affecter des couleurs d'après clusters2 : pour chaque molécule (ID) on assigne la valeur du cluster correspondant
    for cid, mol_ids in clusters2.items():
        for mid in mol_ids:
            colors[mid] = cid

    # Pour chaque cluster dans clusters1, on place le centroid et on répartit les points dans une grille
    for idx, (cid, mol_ids) in enumerate(clusters1.items()):
        cx = (idx // n_side) * s1
        cy = (idx % n_side) * s1
        centroids.append((cx, cy))
        n_in_cluster = len(mol_ids)
        n2 = int(np.ceil(np.sqrt(n_in_cluster)))
        s2 = (s1 - 1) / n2
        sorted_ids = sorted(mol_ids, key=lambda x: colors[x])
        for j, mid in enumerate(sorted_ids):
            x_coord = cx + 1 + (j // n2) * s2
            y_coord = cy + 1 + (j % n2) * s2
            points[mid] = (x_coord, y_coord)
    
    return np.array(centroids), np.array(points), np.array(colors)

def display_comparison(centroids, points, colors, base_marker_size=10, output_path=None):
    """
    Affiche la comparaison des clusters et sauvegarde l'image si output_path est fourni.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'D']
    
    for i in range(len(points)):
        marker = markers[int(colors[i]) % len(markers)] if colors[i] is not None else 'o'
        color = plt.cm.tab10(int(colors[i]) // len(markers)) if colors[i] is not None else 'black'
        ax.plot(points[i][0], points[i][1], marker, color=color, markersize=base_marker_size)
    
    s1 = 10
    for centroid in centroids:
        rect = patches.Rectangle(centroid, s1-1, s1-1, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
    
    ax.set_title("Comparaison des clusters")
    ax.axis("equal")
    if output_path:
        plt.savefig(output_path)
        logger.info("Image saved to %s", output_path)
    plt.show()

def main_compare(cluster_file1: str, cluster_file2: str, output_image: str = None):
    """
    Lit deux fichiers JSON de clustering, compare les clusters et affiche/sauvegarde l'image.
    
    Si output_image n'est pas fourni, il sera généré automatiquement dans
    output/comparison/centroids/ avec un nom combinant les deux noms de fichiers et un suffixe.
    """
    clusters1 = read_cluster_json(cluster_file1)
    clusters2 = read_cluster_json(cluster_file2)
    if total_molecules(clusters1) != total_molecules(clusters2):
        raise ValueError("Les deux fichiers de clustering ne contiennent pas le même nombre de molécules.")
    
    centroids, points, colors = compare_clusters(clusters1, clusters2)
    
    if output_image is None:
        base1 = os.path.splitext(os.path.basename(cluster_file1))[0]
        base2 = os.path.splitext(os.path.basename(cluster_file2))[0]
        output_dir = os.path.join("output", "comparison", "centroids")
        os.makedirs(output_dir, exist_ok=True)
        output_image = os.path.join(output_dir, f"{base1}_{base2}_comparison.png")
    
    display_comparison(centroids, points, colors, base_marker_size=10, output_path=output_image)

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <cluster_file1.json> <cluster_file2.json> [output_image.png]")
        sys.exit(1)
    cluster_file1 = sys.argv[1]
    cluster_file2 = sys.argv[2]
    output_image = sys.argv[3] if len(sys.argv) > 3 else None
    main_compare(cluster_file1, cluster_file2, output_image)
