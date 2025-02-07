#!/usr/bin/env python3
"""
Point d'entrée global pour spectra_analyser.

Utilisation:
  spectra_analyser <command> [options]

Commandes disponibles :
  process         Exécute le parsing et le découpage du fichier MGF (module processing).
  cluster_spectra Exécute le pipeline de clustering kmeans sur un fichier MGF de spectres.
  cluster_smiles  Exécute le pipeline de clustering kmeans sur un fichier de SMILES.
  
Exemples :
  spectra_analyser process --mgf_file data/ALL_GNPS_cleaned.mgf --output_dir data/adducts --stats file --log-level INFO
  spectra_analyser cluster_spectra --mgf_file data/adducts/spectra/[M+H]1+_example.mgf --bin_size 5 --k_min 2 --k_max 10 --algorithm mini --n_init 10 --random_state 42 --mz_min 20 --mz_max 2000 --n_jobs -1 --log-level INFO
  spectra_analyser cluster_smiles --smiles_file data/adducts/smiles/smiles.txt --fp_size 2048 --k_min 2 --k_max 10 --algorithm mini --n_init 10 --random_state 42 --n_jobs -1 --log-level INFO
"""

import argparse
import logging
import config
from processing import mgf_processor
from spectra.clustering_pipeline import kmeans as spectra_kmeans
from smiles.clustering_pipeline import kmeans as smiles_kmeans

__all__ = ["main"]

def main():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--log-level", type=str,
                               choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                               default="INFO",
                               help="Niveau de log (défaut: INFO)")
    
    parser = argparse.ArgumentParser(
        description="spectra_analyser: Analyse et traitement de spectres et SMILES moléculaires.",
        parents=[parent_parser]
    )
    subparsers = parser.add_subparsers(dest="command", help="Commandes disponibles")
    
    # Sous-commande 'process'
    parser_process = subparsers.add_parser("process",
                                             help="Parsage et découpage du fichier MGF",
                                             parents=[parent_parser])
    parser_process.add_argument("--mgf_file", type=str, default=config.DEFAULT_MGF_FILE,
                                help=f"Fichier MGF à traiter (défaut: {config.DEFAULT_MGF_FILE})")
    parser_process.add_argument("--output_dir", type=str, default=config.DEFAULT_OUTPUT_DIR,
                                help=f"Répertoire de sortie (défaut: {config.DEFAULT_OUTPUT_DIR})")
    parser_process.add_argument("--stats", nargs="?", const="console", choices=["console", "file"],
                                help="Si spécifié, calcule les statistiques (console ou file)")
    
    # Sous-commande 'cluster_spectra'
    parser_cluster = subparsers.add_parser("cluster_spectra",
                                             help="Pipeline de clustering kmeans pour spectres (fichier MGF)",
                                             parents=[parent_parser])
    parser_cluster.add_argument("--mgf_file", type=str, required=True,
                                help="Fichier MGF de spectres à traiter.")
    parser_cluster.add_argument("--bin_size", type=float, required=True,
                                help="Taille du bin pour le fixed binning.")
    parser_cluster.add_argument("--k_min", type=int, required=True,
                                help="Nombre minimal de clusters à tester.")
    parser_cluster.add_argument("--k_max", type=int, required=True,
                                help="Nombre maximal de clusters à tester.")
    parser_cluster.add_argument("--algorithm", type=str, choices=["mini", "kmeans"], default="mini",
                                help="Algorithme à utiliser: 'mini' pour MiniBatchKMeans ou 'kmeans' pour KMeans standard (défaut: mini)")
    parser_cluster.add_argument("--n_init", type=int, default=10,
                                help="Nombre d'initialisations (défaut: 10)")
    parser_cluster.add_argument("--random_state", type=int, default=42,
                                help="Graine aléatoire (défaut: 42)")
    parser_cluster.add_argument("--mz_min", type=float, default=20,
                                help="Valeur minimale de m/z (défaut: 20)")
    parser_cluster.add_argument("--mz_max", type=float, default=2000,
                                help="Valeur maximale de m/z (défaut: 2000)")
    parser_cluster.add_argument("--n_jobs", type=int, default=-1,
                                help="Nombre de jobs parallèles pour la sélection de k (défaut: -1)")
    
    # Sous-commande 'cluster_smiles'
    parser_smiles = subparsers.add_parser("cluster_smiles",
                                          help="Pipeline de clustering kmeans pour SMILES",
                                          parents=[parent_parser])
    parser_smiles.add_argument("--smiles_file", type=str, required=True,
                               help="Fichier texte contenant des SMILES (un par ligne).")
    parser_smiles.add_argument("--fp_size", type=int, default=2048,
                               help="Taille du fingerprint Morgan (défaut: 2048)")
    parser_smiles.add_argument("--k_min", type=int, required=True,
                               help="Nombre minimal de clusters à tester.")
    parser_smiles.add_argument("--k_max", type=int, required=True,
                               help="Nombre maximal de clusters à tester.")
    parser_smiles.add_argument("--algorithm", type=str, choices=["mini", "kmeans"], default="mini",
                               help="Algorithme à utiliser (défaut: mini)")
    parser_smiles.add_argument("--n_init", type=int, default=10,
                               help="Nombre d'initialisations (défaut: 10)")
    parser_smiles.add_argument("--random_state", type=int, default=42,
                               help="Graine aléatoire (défaut: 42)")
    parser_smiles.add_argument("--n_jobs", type=int, default=-1,
                               help="Nombre de jobs parallèles pour la sélection de k (défaut: -1)")
    
    args = parser.parse_args()
    
    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format="[%(levelname)s] %(message)s")
    
    if args.command == "process":
        mgf_processor.process_mgf_file(args.mgf_file, args.output_dir, stats_mode=args.stats)
    elif args.command == "cluster_spectra":
        spectra_kmeans.run_clustering_pipeline(
            mgf_file=args.mgf_file,
            bin_size=args.bin_size,
            k_min=args.k_min,
            k_max=args.k_max,
            n_init=args.n_init,
            random_state=args.random_state,
            algorithm=args.algorithm,
            mz_min=args.mz_min,
            mz_max=args.mz_max,
            n_jobs=args.n_jobs
        )
    elif args.command == "cluster_smiles":
        smiles_kmeans.run_clustering_pipeline(
            smiles_file=args.smiles_file,
            fp_size=args.fp_size,
            k_min=args.k_min,
            k_max=args.k_max,
            n_init=args.n_init,
            random_state=args.random_state,
            algorithm=args.algorithm,
            n_jobs=args.n_jobs
        )
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
