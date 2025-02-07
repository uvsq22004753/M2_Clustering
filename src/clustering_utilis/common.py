import json
import hashlib
from datetime import datetime

def generate_hash(metadata: dict) -> str:
    """
    Génère un hash MD5 à partir d'un dictionnaire de paramètres, en excluant les clés variables.
    Retourne une chaîne de 8 caractères.
    """
    filtered = {k: metadata[k] for k in sorted(metadata) if k not in ['date', 'timestamp']}
    metadata_str = json.dumps(filtered, sort_keys=True)
    return hashlib.md5(metadata_str.encode('utf-8')).hexdigest()[:8]

def write_json_results(params, performance, results, output_file):
    """
    Sauvegarde les résultats du clustering dans un fichier JSON avec la structure suivante :
      - metadata: date, paramètres, performance.
      - results: assignations de clusters.
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
