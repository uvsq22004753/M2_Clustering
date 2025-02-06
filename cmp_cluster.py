import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys


def parser_julie(data):
    """
    Transforme les données de clusters de la forme:
        [{'Cluster_ID': int, 'Molecule_ID': int}]
    vers la forme:
        {'Cluster_ID': [int]}
    """
    res = dict()
    for v in data['clusters']:
        res[v['Cluster_ID']] = res.get(v['Cluster_ID'], []) + [int(v['Molecule_ID'])]
    return res


def open_json(file="./clusters.json"):
    """
    Ouvre un fichier json contenant les clusters et le parse suivant sont format.
    Retourne un dict où la clef est le nom du cluster et la valeur une list d'id de molecule.
    """
    with open(file) as fp:
        data =json.load(fp)
    if len(data) == 1: # format Julie
        data = parser_julie(data)
    elif len(data) == 2: # format Mohamed
        data = parser_julie(data['results'])
    else: # format Pauline/Lazare
        #data = data['cluster']
        pass
    return data



def nbr_mol(clusters):
    """
    Retourne le nombre de molecules total dans les clusters.
    """
    s = 0
    for cluster in clusters.values():
        s += len(cluster)
    return s

def compare(clusters1, clusters2):
    """
    Calcule les coordonnées des points et des clusters puis les affiche.
    """
    n1 = np.ceil(np.sqrt(len(clusters1)))
    s1 = 10
    x = nbr_mol(clusters1)
    print(x)
    centroids = []
    points = [None]*x
    colors = [None]*x
    for i, cluster in enumerate(clusters2.values()):
        for mol_id in cluster:
            colors[mol_id] = i

    max_n2 = 0
    for i, cluster in enumerate(clusters1.values()):
        cx, cy = (i//n1)*s1, (i%n1)*s1
        centroids.append((cx, cy))
        n2 = np.ceil(np.sqrt(len(cluster)))
        s2 = (s1-1) / n2
        max_n2 = max(max_n2, n2)
        cluster = sorted(cluster, key=lambda e: colors[e])
        for j, mol_id in enumerate(cluster):
            points[mol_id]=(cx+1+(j//n2)*s2, cy+1+(j%n2)*s2)

    centroids = np.array(centroids)
    points = np.array(points)
    colors = np.array(colors)

    _display(centroids, points, colors, s1/max_n2)


def _display(centroids, points, colors, base_marker_size=5):
    """
    Fonction d'affichage des clusters:
    - centroids, coordonnées des clusters du premier fichier (carrés noirs)
    - points, coordonnées des points représentant les molécules
    - colors, ID des clusters du deuxieme fichier (determine la forme et la couleur)
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    markers = ['o','v','^','<','>','1','2','3','4','s','p','P','*','h','H','+','x','X','D','d','|','_']
    s1=10
    #base_marker_size = s1/np.sqrt()
    
    def get_marker_size():
        """
        Retourne la taille des marqueurs en fonction du niveau de zoom
        """
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        scale_factor = (xlim[1] - xlim[0] + ylim[1] - ylim[0]) / 2
        return max(5, base_marker_size / scale_factor)

    plots = []
    prev_marker_size = get_marker_size()
    for i in range(len(points)):
        plot, = ax.plot(
            points[i][0], points[i][1], 
            markers[colors[i] % len(markers)], 
            c=plt.cm.tab10(colors[i] //len(markers)),
            ms=prev_marker_size
        )
        plots.append(plot)
    
    for centroid in centroids:
        square = patches.Rectangle(centroid, s1-1, s1-1, edgecolor='black', facecolor='none')
        ax.add_patch(square)
    
    def on_zoom(event):
        """
        Ajuste la taille des marqueurs de matplotlib en fonction du niveau de zoom
        """
        nonlocal prev_marker_size

        new_marker_size = get_marker_size()

        if abs(new_marker_size - prev_marker_size) > 0.001:
            print("updating...", prev_marker_size, new_marker_size)
            for plot in plots:
                plot.set_markersize(new_marker_size)
            prev_marker_size = new_marker_size
            fig.canvas.draw()
            

    fig.canvas.mpl_connect("draw_event", on_zoom)

    ax.set_title("Cluster Comparison")
    ax.axis("equal")
    plt.show()



if __name__ == '__main__':
    if len(sys.argv) == 3:
        clusters1 = open_json(sys.argv[1])
        clusters2 = open_json(sys.argv[2])
        #clusters1, clusters2 = filter_clusters(clusters1, clusters2, 3)
        #print(clusters1)
        #print(clusters2)
        if nbr_mol(clusters1) != nbr_mol(clusters2):
            print("[ERROR] Clusters do not have the same amount of molecules id.\nAborting...")
            exit(1)
        compare(clusters1, clusters2)
    else:
        print(f"usage: python3 {sys.argv[0]} CLUSTER_FILE_1 CLUSTER_FILE_2")
