from pyteomics import mgf
import numpy
import os
import sys
import shutil

def _print_help():
    print(f"python3 {sys.argv[0]} [MGF File] [Output Directory]")

def _new_dir(directory):
    if os.path.exists(directory):
        if input(f"[WARNING] Remove '{directory}' directory ? [y/N] ").lower()=='y':
            shutil.rmtree(directory)
        else:
            print("Aborting...")
            exit(0)
    os.mkdir(directory)


def _clean_compound_name(compound_name):
    if "CollisionEnergy" in compound_name:
        tmp = compound_name.rsplit("CollisionEnergy", 1)
        if len(tmp) >= 3:
            print(" [WARNING] +2 CollisionEnergy:", tmp)
        return tmp[0]
    else:
        return compound_name


def _print_dict(dictionary, boundless=False):
    width = 0
    for key, value in dictionary.items():
        width = max(width, len(str(key)))
        

    if not boundless: width = min(width, 50)
    for key, value in dictionary.items():
        if len(key) > width: key = key[:width]
        print(f"{key:<{width}} : {str(value)}")


def _filter_params(params):
    compound_name = params.get('compound_name')
    compound_name = _clean_compound_name(compound_name)
    smiles = params.get('smiles')
    return dict(smiles=smiles, compound_name=compound_name)

def _filter_peaks(mz_array, intensity_array, mz_from=20, mz_to=20000, min_intensity=0.001):
    mask = numpy.logical_and(mz_from<=mz_array, mz_array<=mz_to)
    mz_array = mz_array[mask]
    intensity_array = intensity_array[mask]

    if intensity_array.size == 0:
        return numpy.empty(shape=(0)), numpy.empty(shape=(0))

    max_intensity = numpy.max(intensity_array)
    if max_intensity <= 0:
        return numpy.empty(shape=(0)), numpy.empty(shape=(0))
    intensity_array = intensity_array / numpy.max(intensity_array)

    mask = intensity_array > min_intensity
    mz_array = mz_array[mask]
    intensity_array = intensity_array[mask]

    return mz_array, intensity_array

def _fingerprint(params):
    return chr(30).join([f"{i}{chr(31)}{params[i]}" for i in sorted(params) if i != 'id'])

def fix_and_split_mgf_file(mgf_file="./ALL_GNPS_cleaned.mgf", output_dir="./adducts", stats=False):
    _new_dir(output_dir)
    print(f"Processing {mgf_file} file.")
    print("[INFO] Quietly discarding when no SMILES.")
    spectra_by_adduct = dict()
    id_by_adduct = dict()
    if stats:
        fingerprint_by_adduct = dict()
        smiles_by_adduct = dict()
    total_discarded = 0
    with mgf.read(mgf_file, use_index=False) as spectra:
        spectra_size = 542777
        print(f"{spectra_size} spectra.")
        nbr_digits = len(str(spectra_size))
        for i, spectrum in enumerate(spectra):
            print(f"\r{i:{nbr_digits}}/{spectra_size}", end='')
            params = spectrum.get('params')
            adduct = params.get('adduct')
            params = _filter_params(params)
            smiles = params.get('smiles', '')
            if not smiles:
                # print(" [WARNING] Discarding spectrum: no smiles.")
                total_discarded += 1
                continue

            mz_array = spectrum.get('m/z array')
            intensity_array = spectrum.get('intensity array')

            mz_array, intensity_array = _filter_peaks(mz_array, intensity_array)
            if intensity_array.size == 0:
                print(" [WARNING] Discarding spectrum: no more peaks.")
                total_discarded += 1
                continue
            elif intensity_array.size != mz_array.size:
                print(" [WARNING] Discarding spectrum: filter of peaks went wrong (different sizes).")
                total_discarded += 1
                continue

            params['id'] = id_by_adduct.get(adduct, 0)
            new_spectrum = dict()
            new_spectrum['params'] = params
            new_spectrum['m/z array'] = mz_array
            new_spectrum['intensity array'] = intensity_array
            
            id_by_adduct[adduct] = id_by_adduct.get(adduct, 0) + 1
            spectra_by_adduct[adduct] = spectra_by_adduct.get(adduct, []) + [new_spectrum]
            if stats:
                smiles_by_adduct[adduct] = smiles_by_adduct.get(adduct, set()) | {smiles}
                fingerprint_by_adduct[adduct] = fingerprint_by_adduct.get(adduct, set()) | {_fingerprint(params)}

    print(f"\r{spectra_size}/{spectra_size}")

    print(f"Writing spectra in {output_dir} directory.")
    for adduct in spectra_by_adduct:
        spectra = spectra_by_adduct[adduct]
        if len(spectra) <= 1:
            print(f"[WARNING] Discarded {adduct} adduct, not enough spectra.")
            total_discarded += 1
            continue

        output_file = output_dir + os.sep + adduct + ".mgf"
        with mgf.write(spectra, output_file) as writer:
            nbr_spectra = len(spectra)
            if stats:
                nbr_duplicates = nbr_spectra - len(fingerprint_by_adduct[adduct])
                nbr_smiles = len(smiles_by_adduct[adduct])
                print(f"Adduct {adduct:20} saved: {nbr_smiles:{nbr_digits}} smiles | {nbr_duplicates:{nbr_digits}} duplicates | {nbr_spectra:{nbr_digits}} spectra")
            else:
                print(f"Adduct {adduct:20} saved: {nbr_spectra:{nbr_digits}} spectra")

    print(f"Total number of spectra discarded: {total_discarded}.")


if __name__ == '__main__':
    match len(sys.argv):
        case 1:
            fix_and_split_mgf_file()
        case 2:
            if "help" in sys.argv[1].lower():
                _print_help()
            else:
                fix_and_split_mgf_file(sys.argv[1])
        case 3:
            fix_and_split_mgf_file(sys.argv[1], sys.argv[2])
        case _:
            print("[ERROR] Too many arguments.")
            _print_help()
