#!/usr/bin/env python3
"""
Point d'entrée global pour spectra_analyser.

Utilisation:
  spectra_analyser <command> [options]

Commandes disponibles :
  process            Exécute le parsing et le découpage du fichier MGF (module processing).
  kmeans_spectra     Pipeline de clustering kmeans pour spectres (fichier MGF).
  kmeans_smiles      Pipeline de clustering kmeans pour SMILES.
  hac_spectra        Pipeline de clustering HAC pour spectres (fichier MGF).
  hac_smiles         Pipeline de clustering HAC pour SMILES.
  hdbscan_spectra    Pipeline de clustering HDBSCAN pour spectres (fichier MGF).
  hdbscan_smiles     Pipeline de clustering HDBSCAN pour SMILES.
  compare_clusters   Compare deux fichiers JSON de clustering et sauvegarde l'image de la comparaison.
  compare_scores     Compare deux fichiers JSON de clustering et affiche les scores ARI et NMI.
  


Exemples :
  python cli.py process --mgf_file data/ALL_GNPS_cleaned.mgf --output_dir data/adducts --stats file --log-level INFO
  python cli.py kmeans_spectra --mgf_file data/adducts/spectra/[M-3H2O+H]1+.mgf --bin_size 5 --k_min 2 --k_max 10 --algorithm mini --n_init 10 --random_state 42 --mz_min 20 --mz_max 2000 --n_jobs -1 --log-level INFO
  python cli.py kmeans_smiles --smiles_file data/adducts/smiles/[M-3H2O+H]1+.smiles --fp_size 2048 --k_min 2 --k_max 10 --algorithm mini --n_init 10 --random_state 42 --n_jobs -1 --log-level INFO
  python cli.py hac_spectra --mgf_file data/adducts/spectra/[M-3H2O+H]1+.mgf --bin_size 5 --n_clusters 4 --mz_min 20 --mz_max 2000 --tol 0.1 --dist_method cosine_greedy --num_workers -1 --log-level INFO
  python cli.py hac_smiles --smiles_file data/adducts/smiles/[M-3H2O+H]1+.smiles --fp_size 2048 --n_clusters 4 --sim_type cosinus --log-level INFO
  python cli.py hdbscan_spectra --mgf_file data/adducts/spectra/[M-3H2O+H]1+.mgf --bin_size 5 --n_clusters 4 --min_samples 2 --mz_min 20 --mz_max 2000 --tol 0.1 --dist_method cosine_greedy --num_workers -1 --log-level INFO
  python cli.py hdbscan_smiles --smiles_file data/adducts/smiles/[M-3H2O+H]1+.smiles --fp_size 2048 --n_clusters 4 --min_samples 1 --sim_type cosinus --log-level INFO
  python cli.py compare_clusters --cluster_file1 output\clustering_results\hac\smiles\[M-3H2O+H]1+_fp2048\[M-3H2O+H]1+_hac_18_4a92a3a1.json  --cluster_file2 .\output\clustering_results\hac\spectra\[M-3H2O+H]1+_Bin5.0\[M-3H2O+H]1+_hac_18_5485cd75.json --log-level INFO
  python cli.py compare_scores --cluster_file1 output\clustering_results\hac\smiles\[M-3H2O+H]1+_fp2048\[M-3H2O+H]1+_hac_18_4a92a3a1.json  --cluster_file2 .\output\clustering_results\hac\spectra\[M-3H2O+H]1+_Bin5.0\[M-3H2O+H]1+_hac_18_5485cd75.json
"""

import argparse
import logging
import config
from processing import mgf_processor
from spectra.clustering_pipeline import kmeans as spectra_kmeans
from spectra.clustering_pipeline import hac as spectra_hac
from smiles.clustering_pipeline import kmeans as smiles_kmeans
from smiles.clustering_pipeline import hac as smiles_hac
from smiles.clustering_pipeline import hdbscan as smiles_hdbscan
from cluster_comparison import compare as comp
from cluster_comparison import scores as comp_scores


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
    
    # Commande 'process'
    parser_process = subparsers.add_parser("process",
                                             help="Parsage et découpage du fichier MGF",
                                             parents=[parent_parser])
    parser_process.add_argument("--mgf_file", type=str, default=config.DEFAULT_MGF_FILE,
                                help=f"Fichier MGF à traiter (défaut: {config.DEFAULT_MGF_FILE})")
    parser_process.add_argument("--output_dir", type=str, default=config.DEFAULT_OUTPUT_DIR,
                                help=f"Répertoire de sortie (défaut: {config.DEFAULT_OUTPUT_DIR})")
    parser_process.add_argument("--stats", nargs="?", const="console", choices=["console", "file"],
                                help="Si spécifié, calcule les statistiques (console ou file)")
    
    # Commande 'kmeans_spectra'
    parser_kmeans_spec = subparsers.add_parser("kmeans_spectra",
                                                 help="Pipeline de clustering kmeans pour spectres (fichier MGF)",
                                                 parents=[parent_parser])
    parser_kmeans_spec.add_argument("--mgf_file", type=str, required=True,
                                    help="Fichier MGF de spectres à traiter.")
    parser_kmeans_spec.add_argument("--bin_size", type=float, required=True,
                                    help="Taille du bin pour le fixed binning.")
    parser_kmeans_spec.add_argument("--k_min", type=int, required=True,
                                    help="Nombre minimal de clusters à tester.")
    parser_kmeans_spec.add_argument("--k_max", type=int, required=True,
                                    help="Nombre maximal de clusters à tester.")
    parser_kmeans_spec.add_argument("--algorithm", type=str, choices=["mini", "kmeans"], default="mini",
                                    help="Algorithme à utiliser (défaut: mini)")
    parser_kmeans_spec.add_argument("--n_init", type=int, default=10,
                                    help="Nombre d'initialisations (défaut: 10)")
    parser_kmeans_spec.add_argument("--random_state", type=int, default=42,
                                    help="Graine aléatoire (défaut: 42)")
    parser_kmeans_spec.add_argument("--mz_min", type=float, default=20,
                                    help="Valeur minimale de m/z (défaut: 20)")
    parser_kmeans_spec.add_argument("--mz_max", type=float, default=2000,
                                    help="Valeur maximale de m/z (défaut: 2000)")
    parser_kmeans_spec.add_argument("--n_jobs", type=int, default=-1,
                                    help="Nombre de jobs parallèles (défaut: -1)")
    
    # Commande 'kmeans_smiles'
    parser_kmeans_smiles = subparsers.add_parser("kmeans_smiles",
                                                 help="Pipeline de clustering kmeans pour SMILES",
                                                 parents=[parent_parser])
    parser_kmeans_smiles.add_argument("--smiles_file", type=str, required=True,
                                      help="Fichier texte contenant des SMILES (un par ligne).")
    parser_kmeans_smiles.add_argument("--fp_size", type=int, default=2048,
                                      help="Taille du fingerprint Morgan (défaut: 2048)")
    parser_kmeans_smiles.add_argument("--k_min", type=int, required=True,
                                      help="Nombre minimal de clusters à tester.")
    parser_kmeans_smiles.add_argument("--k_max", type=int, required=True,
                                      help="Nombre maximal de clusters à tester.")
    parser_kmeans_smiles.add_argument("--algorithm", type=str, choices=["mini", "kmeans"], default="mini",
                                      help="Algorithme à utiliser (défaut: mini)")
    parser_kmeans_smiles.add_argument("--n_init", type=int, default=10,
                                      help="Nombre d'initialisations (défaut: 10)")
    parser_kmeans_smiles.add_argument("--random_state", type=int, default=42,
                                      help="Graine aléatoire (défaut: 42)")
    parser_kmeans_smiles.add_argument("--n_jobs", type=int, default=-1,
                                      help="Nombre de jobs parallèles (défaut: -1)")
    
    # Commande 'hac_spectra'
    parser_hac_spec = subparsers.add_parser("hac_spectra",
                                             help="Pipeline de clustering HAC pour spectres (fichier MGF)",
                                             parents=[parent_parser])
    parser_hac_spec.add_argument("--mgf_file", type=str, required=True,
                                 help="Fichier MGF de spectres à traiter.")
    parser_hac_spec.add_argument("--bin_size", type=float, required=True,
                                 help="Taille du bin pour le binning.")
    parser_hac_spec.add_argument("--n_clusters", type=int, required=True,
                                 help="Nombre de clusters à former avec HAC.")
    parser_hac_spec.add_argument("--mz_min", type=float, default=20,
                                 help="Valeur minimale de m/z (défaut: 20)")
    parser_hac_spec.add_argument("--mz_max", type=float, default=2000,
                                 help="Valeur maximale de m/z (défaut: 2000)")
    parser_hac_spec.add_argument("--tol", type=float, default=0.1,
                                 help="Tolérance pour le calcul de la matrice de distance (défaut: 0.1)")
    parser_hac_spec.add_argument("--num_workers", type=int, default=-1,
                                 help="Nombre de workers pour le calcul parallèle (défaut: -1)")
    parser_hac_spec.add_argument("--dist_method", type=str,
                                 choices=["cosinus", "manhattan", "simple", "cosine_greedy"],
                                 default="cosinus",
                                 help="Méthode de calcul de distance pour les spectres (défaut: cosinus)")
    
    # Commande 'hac_smiles'
    parser_hac_smiles = subparsers.add_parser("hac_smiles",
                                              help="Pipeline de clustering HAC pour SMILES",
                                              parents=[parent_parser])
    parser_hac_smiles.add_argument("--smiles_file", type=str, required=True,
                                   help="Fichier texte contenant des SMILES (un par ligne).")
    parser_hac_smiles.add_argument("--fp_size", type=int, default=2048,
                                   help="Taille du fingerprint Morgan (défaut: 2048)")
    parser_hac_smiles.add_argument("--n_clusters", type=int, required=True,
                                   help="Nombre de clusters à former avec HAC.")
    parser_hac_smiles.add_argument("--sim_type", type=str,
                                   choices=["cosinus", "jaccard"],
                                   default="jaccard",
                                   help="Type de similarité pour SMILES (défaut: jaccard)")
    
    # Commande 'hdbscan_spectra'
    parser_hdbscan_spec = subparsers.add_parser("hdbscan_spectra",
                                                help="Pipeline de clustering HDBSCAN pour spectres (fichier MGF)",
                                                parents=[parent_parser])
    parser_hdbscan_spec.add_argument("--mgf_file", type=str, required=True,
                                     help="Fichier MGF de spectres à traiter.")
    parser_hdbscan_spec.add_argument("--bin_size", type=float, required=True,
                                     help="Taille du bin pour le binning.")
    parser_hdbscan_spec.add_argument("--n_clusters", type=int, required=True,
                                     help="Nombre de clusters à former (utilisé comme min_cluster_size).")
    parser_hdbscan_spec.add_argument("--min_samples", type=int, required=True,
                                     help="Nombre minimum d'échantillons pour HDBSCAN.")
    parser_hdbscan_spec.add_argument("--mz_min", type=float, default=20,
                                     help="Valeur minimale de m/z (défaut: 20)")
    parser_hdbscan_spec.add_argument("--mz_max", type=float, default=2000,
                                     help="Valeur maximale de m/z (défaut: 2000)")
    parser_hdbscan_spec.add_argument("--tol", type=float, default=0.1,
                                     help="Tolérance pour le calcul de la matrice de distance (défaut: 0.1)")
    parser_hdbscan_spec.add_argument("--num_workers", type=int, default=-1,
                                     help="Nombre de workers pour le calcul parallèle (défaut: -1)")
    parser_hdbscan_spec.add_argument("--dist_method", type=str,
                                     choices=["cosinus", "manhattan", "simple", "cosine_greedy"],
                                     default="cosinus",
                                     help="Méthode de calcul de distance pour les spectres (défaut: cosinus)")
    
    # Commande 'hdbscan_smiles'
    parser_hdbscan_smiles = subparsers.add_parser("hdbscan_smiles",
                                                  help="Pipeline de clustering HDBSCAN pour SMILES",
                                                  parents=[parent_parser])
    parser_hdbscan_smiles.add_argument("--smiles_file", type=str, required=True,
                                      help="Fichier texte contenant des SMILES (un par ligne).")
    parser_hdbscan_smiles.add_argument("--fp_size", type=int, default=2048,
                                      help="Taille du fingerprint Morgan (défaut: 2048)")
    parser_hdbscan_smiles.add_argument("--n_clusters", type=int, required=True,
                                      help="Nombre de clusters à former (utilisé comme min_cluster_size).")
    parser_hdbscan_smiles.add_argument("--min_samples", type=int, default=1,
                                      help="Nombre minimum d'échantillons pour HDBSCAN (défaut: 1)")
    parser_hdbscan_smiles.add_argument("--sim_type", type=str,
                                      choices=["cosinus", "jaccard"],
                                      default="jaccard",
                                      help="Type de similarité pour SMILES (défaut: jaccard)")
    
    # commande 'compare_clusters'
    parser_compare = subparsers.add_parser("compare_clusters",
                                            help="Compare deux fichiers JSON de clustering et sauvegarde une image des centroids",
                                            parents=[parent_parser])
    parser_compare.add_argument("--cluster_file1", type=str, required=True,
                                help="Premier fichier JSON de clustering.")
    parser_compare.add_argument("--cluster_file2", type=str, required=True,
                                help="Deuxième fichier JSON de clustering.")
    parser_compare.add_argument("--output_image", type=str, default=None,
                                help="Chemin du fichier image de sortie. Si non spécifié, il sera généré automatiquement dans output/comparison/centroids/")

    # Commande 'compare_scores'
    parser_compare = subparsers.add_parser("compare_scores",
                                            help="Compare deux fichiers JSON de clustering et affiche les scores ARI et NMI",
                                            parents=[parent_parser])
    parser_compare.add_argument("--cluster_file1", type=str, required=True,
                                help="Premier fichier JSON de clustering.")
    parser_compare.add_argument("--cluster_file2", type=str, required=True,
                                help="Deuxième fichier JSON de clustering.")

        
    
    args = parser.parse_args()
    
    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format="[%(levelname)s] %(message)s")
    
    if args.command == "process":
        mgf_processor.process_mgf_file(args.mgf_file, args.output_dir, stats_mode=args.stats)
    elif args.command == "kmeans_spectra":
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
    elif args.command == "kmeans_smiles":
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
    elif args.command == "hac_spectra":
        num_workers = args.num_workers if args.num_workers != -1 else None
        from spectra.clustering_pipeline import hac as spectra_hac
        spectra_hac.run_hac_pipeline(
            mgf_file=args.mgf_file,
            bin_size=args.bin_size,
            n_clusters=args.n_clusters,
            opt="somme",  # On utilise "somme" pour le binning ici
            mz_min=args.mz_min,
            mz_max=args.mz_max,
            tol=args.tol,
            num_workers=num_workers,
            dist_method=args.dist_method
        )
    elif args.command == "hac_smiles":
        from smiles.clustering_pipeline import hac as smiles_hac
        smiles_hac.run_hac_pipeline_smiles(
            smiles_file=args.smiles_file,
            fp_size=args.fp_size,
            k_clusters=args.n_clusters,
            sim_type=args.sim_type
        )
    elif args.command == "hdbscan_spectra":
        num_workers = args.num_workers if args.num_workers != -1 else None
        from spectra.clustering_pipeline import hdbscan as spectra_hdbscan
        spectra_hdbscan.run_hdbscan_pipeline(
            mgf_file=args.mgf_file,
            bin_size=args.bin_size,
            n_clusters=args.n_clusters,
            min_samples=args.min_samples,
            mz_min=args.mz_min,
            mz_max=args.mz_max,
            tol=args.tol,
            num_workers=num_workers,
            dist_method=args.dist_method
        )
    elif args.command == "hdbscan_smiles":
      smiles_hdbscan.run_hdbscan_pipeline_smiles(
          smiles_file=args.smiles_file,
          fp_size=args.fp_size,
          min_cluster_size=args.n_clusters,
          min_samples=args.min_samples,
          sim_type=args.sim_type
      )
    elif args.command == "compare_clusters":
      comp.main_compare(args.cluster_file1, args.cluster_file2, args.output_image)
    elif args.command == "compare_scores":
      comp_scores.compare_clusterings(args.cluster_file1, args.cluster_file2)

    else:
        parser.print_help()

if __name__ == '__main__':
    main()
