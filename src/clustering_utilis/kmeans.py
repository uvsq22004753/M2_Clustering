import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from joblib import Parallel, delayed
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

def normalize_features(X):
    """
    Normalise chaque ligne (échantillon) de X par sa norme L2.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1  # éviter la division par zéro
    return X / norms

def evaluate_k(X, k, n_init, random_state, algorithm='mini'):
    """
    Évalue une valeur de k en exécutant KMeans (ou MiniBatchKMeans) sur X
    et en calculant le score de silhouette.
    Si le clustering ne forme qu'un seul cluster, renvoie un score de -1.
    """
    if algorithm == 'mini':
        model = MiniBatchKMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    else:
        model = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    labels = model.fit_predict(X)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        score = -1  # Pas assez de clusters pour calculer la silhouette
        logger.info("Pour k = %d, un seul cluster a été formé (labels: %s).", k, unique_labels)
    else:
        score = silhouette_score(X, labels)
        logger.info("Pour k = %d, silhouette score = %.4f", k, score)
    return k, score

def select_best_k(X, k_min, k_max, n_init=10, random_state=42, algorithm='mini', n_jobs=-1):
    """
    Teste en parallèle les valeurs de k de k_min à k_max et retourne le meilleur k basé sur le score de silhouette.
    
    Avant de lancer les tests, cette fonction vérifie que :
      - Il y a au moins 2 échantillons.
      - k_min est au moins 2.
      - k_max ne dépasse pas (n_samples - 1). Sinon, k_max est ajusté.
    
    Retourne :
      - best_k : le meilleur k sélectionné.
      - scores : dictionnaire des scores pour chaque k testé.
    """
    n_samples = X.shape[0]
    if n_samples < 2:
        raise ValueError("Le nombre d'échantillons doit être au moins 2 pour effectuer le clustering.")
    if k_min < 2:
        logger.warning("k_min ajusté à 2 car le clustering nécessite au moins 2 clusters.")
        k_min = 2
    if k_max >= n_samples:
        logger.warning("k_max ajusté à %d car il ne peut excéder n_samples", n_samples)
        k_max = n_samples

    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_k)(X, k, n_init, random_state, algorithm) for k in range(k_min, k_max + 1)
    )
    scores = {k: score for k, score in results}
    best_k = max(scores, key=scores.get)
    logger.info("Meilleur k sélectionné : %d avec un score de silhouette de %.4f", best_k, scores[best_k])
    return best_k, scores

def run_kmeans(X, k, n_init=10, random_state=42, algorithm='mini'):
    """
    Exécute KMeans (ou MiniBatchKMeans) sur X avec k clusters et retourne les labels, centres et le score de silhouette.
    """
    if algorithm == 'mini':
        model = MiniBatchKMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    else:
        model = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    labels = model.fit_predict(X)
    centers = model.cluster_centers_
    score = silhouette_score(X, labels)
    return labels, centers, score

def write_json_results(params, performance, results, output_file):
    """
    Écrit les résultats du clustering dans un fichier JSON avec la structure suivante :
      - metadata (date, paramètres, performance)
      - results (liste d'assignations)
    """
    output = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "parameters": params,
            "performance": performance
        },
        "results": results
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=4)
    logger.info("Results written to %s", output_file)