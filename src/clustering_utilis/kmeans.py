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

def evaluate_k(X, k, n_init=10, random_state=42, algorithm='mini'):
    """
    Exécute KMeans (ou MiniBatchKMeans) pour un certain nombre de clusters et retourne le score de silhouette.
    """
    if algorithm == 'mini':
        model = MiniBatchKMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    else:
        model = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    logger.info("k = %d, silhouette score = %.4f", k, score)
    return k, score

def select_best_k(X, k_min, k_max, n_init=10, random_state=42, algorithm='mini', n_jobs=-1):
    """
    Teste les valeurs de k de k_min à k_max et retourne le meilleur k basé sur le score de silhouette.
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_k)(X, k, n_init, random_state, algorithm) for k in range(k_min, k_max + 1)
    )
    scores = {k: score for k, score in results}
    best_k = max(scores, key=scores.get)
    logger.info("Best k: %d with silhouette score: %.4f", best_k, scores[best_k])
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