#!/usr/bin/env python3
"""
Point d'entrée global pour spectra_analyser.

Utilisation:
  spectra_analyser process [options]

Commande disponible :
  process      Exécute le parsing et le découpage du fichier MGF (module processing).

Exemples :
  spectra_analyser process --mgf_file data/ALL_GNPS_cleaned.mgf --output_dir data/adducts --stats file --log-level INFO
"""

import argparse
import logging
import config
from processing import mgf_processor

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
    parser_process = subparsers.add_parser("process",
                                            help="Parsage et découpage du fichier MGF",
                                            parents=[parent_parser])
    parser_process.add_argument("--mgf_file", type=str, default=config.DEFAULT_MGF_FILE,
                                help=f"Fichier MGF à traiter (défaut: {config.DEFAULT_MGF_FILE})")
    parser_process.add_argument("--output_dir", type=str, default=config.DEFAULT_OUTPUT_DIR,
                                help=f"Répertoire de sortie (défaut: {config.DEFAULT_OUTPUT_DIR})")
    parser_process.add_argument("--stats", nargs="?", const="console", choices=["console", "file"],
                                help=("Si spécifié, calcule les statistiques. "
                                      "Valeur par défaut 'console' ou 'file' pour sauvegarde."))

    args = parser.parse_args()

    # Configuration du log
    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format="[%(levelname)s] %(message)s")

    if args.command == "process":
        mgf_processor.process_mgf_file(args.mgf_file, args.output_dir, stats_mode=args.stats)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
