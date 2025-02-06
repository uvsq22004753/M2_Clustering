# src/processing/mgf_processor.py
import os
import logging
from pyteomics import mgf
import config
from processing import filters, io

def process_mgf_file(mgf_file: str, output_dir: str, stats_mode: str = None):
    """
    Lit un fichier MGF, applique les filtres sur les spectra et répartit le résultat par adduit.
    
    Pour chaque adduit, crée :
      - Un fichier MGF contenant les spectra filtrées (dans le sous-dossier spectra).
      - Un fichier SMILES listant, pour chaque spectrum, le SMILES correspondant (dans le sous-dossier smiles).
    """
    # Création du répertoire de sortie et de ses sous-dossiers
    io.new_dir(output_dir)
    spectra_output_dir = os.path.join(output_dir, config.SPECTRA_SUBDIR)
    smiles_output_dir = os.path.join(output_dir, config.SMILES_SUBDIR)
    io.new_dir(spectra_output_dir)
    io.new_dir(smiles_output_dir)

    logging.info(f"Processing file: {mgf_file}")
    logging.info("Spectra without SMILES will be discarded silently.")

    spectra_by_adduct = {}
    id_by_adduct = {}
    if stats_mode:
        fingerprint_by_adduct = {}
        smiles_by_adduct = {}
        stats_lines = []  # Accumulation des lignes de statistiques

    total_discarded = 0

    with mgf.read(mgf_file, use_index=False) as spectra:
        for i, spectrum in enumerate(spectra):
            if i >= config.MAX_SPECTRA:
                break
            if i % 1000 == 0:
                logging.info(f"Processing spectrum {i}/{config.SPECTRA_SIZE}")

            params = spectrum.get('params', {})
            adduct = params.get('adduct')
            params = filters.filter_params(params)
            smiles = params.get('smiles', '')
            if not smiles:
                total_discarded += 1
                continue

            mz_array = spectrum.get('m/z array')
            intensity_array = spectrum.get('intensity array')
            mz_array, intensity_array = filters.filter_peaks(
                mz_array, intensity_array,
                mz_from=config.MZ_FROM,
                mz_to=config.MZ_TO,
                min_intensity=config.MIN_INTENSITY
            )

            if intensity_array.size == 0:
                logging.warning("Discarding spectrum: no remaining peaks after filtering.")
                total_discarded += 1
                continue
            if intensity_array.size != mz_array.size:
                logging.warning("Discarding spectrum: mismatch between m/z and intensity array sizes.")
                total_discarded += 1
                continue

            # Attribution d'un ID pour le spectrum dans cet adduit
            params['id'] = id_by_adduct.get(adduct, 0)
            new_spectrum = {
                'params': params,
                'm/z array': mz_array,
                'intensity array': intensity_array
            }
            id_by_adduct[adduct] = id_by_adduct.get(adduct, 0) + 1
            spectra_by_adduct.setdefault(adduct, []).append(new_spectrum)

            if stats_mode:
                smiles_by_adduct.setdefault(adduct, set()).add(smiles)
                fingerprint_by_adduct.setdefault(adduct, set()).add(filters.fingerprint(params))

    logging.info("Finished processing spectra.")
    logging.info(f"Writing output files in '{output_dir}' directory.")

    for adduct, spectra in spectra_by_adduct.items():
        if len(spectra) <= 1:
            logging.warning(f"Discarded adduct '{adduct}': not enough spectra.")
            total_discarded += 1
            continue

        # Écriture du fichier MGF pour cet adduct dans le sous-dossier spectra
        output_mgf = os.path.join(spectra_output_dir, f"{adduct}.mgf")
        io.write_mgf_file(spectra, output_mgf, mgf)

        # Écriture du fichier SMILES pour cet adduct dans le sous-dossier smiles
        output_smiles = os.path.join(smiles_output_dir, f"{adduct}.smiles")
        io.write_smiles_file(spectra, output_smiles)

        if stats_mode:
            nbr_spectra = len(spectra)
            nbr_duplicates = nbr_spectra - len(fingerprint_by_adduct.get(adduct, []))
            nbr_smiles = len(smiles_by_adduct.get(adduct, []))
            line = (f"Adduct {adduct:20} saved: {nbr_smiles:6} smiles | "
                    f"{nbr_duplicates:6} duplicates | {nbr_spectra:6} spectra")
            logging.info(line)
            stats_lines.append(line)
        else:
            logging.info(f"Adduct {adduct:20} saved: {len(spectra):6} spectra")

    logging.info(f"Total number of spectra discarded: {total_discarded}")

    # Sauvegarde des statistiques si demandé en mode "file"
    if stats_mode == "file":
        stats_file_path = os.path.join(output_dir, config.DEFAULT_STATS_FILE)
        try:
            with open(stats_file_path, "w") as f:
                f.write("\n".join(stats_lines))
            logging.info(f"Statistics saved in '{stats_file_path}'")
        except Exception as e:
            logging.error(f"Error saving statistics to file: {e}")
