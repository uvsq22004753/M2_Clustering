import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys


def parser_julie(data):
    res = dict()
    for v in data['clusters']:
        res[v['Cluster_ID']] = res.get(v['Cluster_ID'], []) + [int(v['Molecule_ID'])]
    return res


def open_json(file="./clusters.json"):
    with open(file) as fp:
        data =json.load(fp)
    if len(data) == 1: # format Julie
        data = parser_julie(data)
    elif len(data) == 2: # format Mohamed
        data = parser_julie(data['results'])
    else: # format Pauline/Lazare
        data = data["clusters"]
    return data



def filter_clusters(clusters1, clusters2, s):
    x = [i for cluster in clusters1.values() for i in cluster if len(cluster) > s]
    y = [i for cluster in clusters2.values() for i in cluster if len(cluster) > s and i in x]

    c1 = dict()
    for label in clusters1:
        ids = []
        for mol_id in clusters1[label]:
            if mol_id in y:
                ids.append(y.index(mol_id))
        if ids: c1[label] = ids
    c2 = dict()
    for label in clusters2:
        ids = []
        for mol_id in clusters2[label]:
            if mol_id in y:
                ids.append(y.index(mol_id))
        if ids: c2[label] = ids
    return c1, c2


def nbr_mol(clusters):
    s = 0
    for cluster in clusters.values():
        s += len(cluster)
    return s


def compare(clusters1, clusters2):
    n1 = np.ceil(np.sqrt(len(clusters1)))
    s1 = 10
    x = nbr_mol(clusters1)
    centroids = []
    points = [None]*x
    colors = [None]*x
    for i, cluster in enumerate(clusters1.values()):
        cx, cy = (i//n1)*s1, (i%n1)*s1
        centroids.append((cx, cy))
        n2 = np.ceil(np.sqrt(len(cluster)))
        s2 = (s1-1) / n2
        for j, mol_id in enumerate(cluster):
            points[mol_id]=(cx+0.5+(j//n2)*s2, cy+0.5+(j%n2)*s2)

    for i, cluster in enumerate(clusters2.values()):
        for mol_id in cluster:
            colors[mol_id] = i

    centroids = np.array(centroids)
    points = np.array(points)
    colors = np.array(colors)

    display(centroids, points, colors)

def display(centroids, points, colors):
    plt.figure(figsize=(10, 10))
    markers = ['o','v','^','<','>','1','2','3','4','s','p','P','*','h','H','+','x','X','D','d','|','_']
    for i in range(x):
        plt.plot(points[i][0], points[i][1], markers[i%len(markers)], c=plt.cm.tab10(colors[i]//len(markers)), ms=(3 / ax.get_window_extent().width))

    for i, centroid in enumerate(centroids):
        square = patches.Rectangle(centroid, s1-1, s1-1, edgecolor='black', facecolor='none')
        plt.gca().add_patch(square)

    plt.title("Cluster Comparison")
    plt.legend()
    plt.axis("equal")
    plt.show()


def display(centroids, points, colors):
    fig, ax = plt.subplots(figsize=(10, 10))
    markers = ['o','v','^','<','>','1','2','3','4','s','p','P','*','h','H','+','x','X','D','d','|','_']
    s1=10
    base_marker_size = 2000/np.sqrt(len(centroids))
    
    def get_marker_size():
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        scale_factor = (xlim[1] - xlim[0] + ylim[1] - ylim[0]) / 2
        return max(5, base_marker_size / scale_factor)

    plots = []
    prev_marker_size = get_marker_size()
    for i in range(len(points)):
        plot, = ax.plot(
            points[i][0], points[i][1], 
            markers[i % len(markers)], 
            c=plt.cm.tab10(colors[i] % 10),
            ms=prev_marker_size
        )
        plots.append(plot)
    
    for centroid in centroids:
        square = patches.Rectangle(centroid, s1-1, s1-1, edgecolor='black', facecolor='none')
        ax.add_patch(square)
    
    def on_zoom(event):
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
        print(clusters1)
        print(clusters2)
        if nbr_mol(clusters1) != nbr_mol(clusters2):
            print("[ERROR] Clusters do not have the same amount of molecules id.\nAborting...")
            exit(1)
        compare(clusters1, clusters2)
    else:
        print(f"usage: python3 {sys.argv[0]} CLUSTER_FILE_1 CLUSTER_FILE_2")
