#!/usr/bin/env python3
"""
Point d'entrée global pour spectra_analyser.

Utilisation:
  spectra_analyser <command> [options]

Commandes disponibles :
  process      Exécute le parsing et le découpage du fichier MGF (module processing).
  similarity   Effectue le binning sur des fichiers de spectres et calcule les matrices de similarité.

Exemples :
  spectra_analyser process --mgf_file data/ALL_GNPS_cleaned.mgf --output_dir data/adducts --stats file --log-level INFO
  spectra_analyser similarity --bin_size 5 --method cosinus --log-level INFO
"""

import argparse
import logging
import config
from processing import mgf_processor
from spectra.similarity import matrix, binning

__all__ = ["main"]

def main():
    # Création d'un parser parent pour les options globales
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--log-level", type=str,
                               choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                               default="INFO",
                               help="Niveau de log (défaut: INFO)")

    # Création du parser principal en utilisant le parent parser
    parser = argparse.ArgumentParser(
        description="spectra_analyser: Analyse et traitement de spectres moléculaires.",
        parents=[parent_parser]
    )
    subparsers = parser.add_subparsers(dest="command", help="Commandes disponibles")

    # Sous-commande 'process'
    parser_process = subparsers.add_parser("process", help="Parsage et découpage du fichier MGF", parents=[parent_parser])
    parser_process.add_argument("--mgf_file", type=str, default=config.DEFAULT_MGF_FILE,
                                help=f"Fichier MGF à traiter (défaut: {config.DEFAULT_MGF_FILE})")
    parser_process.add_argument("--output_dir", type=str, default=config.DEFAULT_OUTPUT_DIR,
                                help=f"Répertoire de sortie (défaut: {config.DEFAULT_OUTPUT_DIR})")
    parser_process.add_argument("--stats", nargs="?", const="console", choices=["console", "file"],
                                help=("Si spécifié, calcule les statistiques. "
                                      "Valeur par défaut 'console' ou 'file' pour sauvegarde."))

    # Sous-commande 'similarity'
    parser_similarity = subparsers.add_parser("similarity", help="Effectue le binning et calcule les matrices de similarité", parents=[parent_parser])
    parser_similarity.add_argument("--input_dir", type=str, default=config.DEFAULT_PROCESSED_SPECTRA_DIR,
                                   help=f"Dossier contenant les fichiers MGF de spectres (défaut: {config.DEFAULT_PROCESSED_SPECTRA_DIR})")
    parser_similarity.add_argument("--bin_size", type=float, default=1,
                                   help="Taille du bin pour le binning (défaut: 1)")
    parser_similarity.add_argument("--method", type=str, choices=["cosinus", "manhattan", "simple"],
                                   required=True, help="Méthode de similarité à utiliser")
    parser_similarity.add_argument("--tolerance", type=float, default=0.1,
                                   help="Tolérance pour la méthode 'simple' (défaut: 0.1)")
    parser_similarity.add_argument("--output_dir", type=str, default=config.DEFAULT_SIMILARITY_OUTPUT_DIR,
                                   help=f"Dossier de sortie pour les matrices de similarité (défaut: {config.DEFAULT_SIMILARITY_OUTPUT_DIR})")
    parser_similarity.add_argument("--num_workers", type=int, default=None,
                                   help="Nombre de processus parallèles (défaut: cpu_count//2)")
    parser_similarity.add_argument("--tmp_dir_base", type=str, default=config.DEFAULT_BINNED_TMP_DIR_BASE,
                                   help=f"Dossier de base pour les fichiers binned (défaut: {config.DEFAULT_BINNED_TMP_DIR_BASE})")

    args = parser.parse_args()

    # Configuration du log
    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format="[%(levelname)s] %(message)s")

    if args.command == "process":
        mgf_processor.process_mgf_file(args.mgf_file, args.output_dir, stats_mode=args.stats)
    elif args.command == "similarity":
        # Définir le dossier temporaire pour les fichiers binned
        tmp_dir = f"{args.tmp_dir_base}_{str(args.bin_size)}"
        logging.info(f"Binning: input={args.input_dir} ; bin_size={args.bin_size} ; output={tmp_dir}")
        from spectra.similarity import binning as sim_binning
        sim_binning.all_mass_spectra_binning(args.input_dir, tmp_dir, bin_size=args.bin_size, opt='somme')
        # Calcul des matrices de similarité depuis les fichiers binned
        logging.info(f"Calcul de matrices de similarité depuis les fichiers binned dans {tmp_dir}")
        from spectra.similarity import matrix
        matrix.make_matrices_from_dir(tmp_dir, args.method, args.output_dir, tol=args.tolerance, num_workers=args.num_workers, bin_size=args.bin_size)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
