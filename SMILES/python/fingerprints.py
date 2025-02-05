from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator, MACCSkeys
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from smiles_utils import readfile_without_cn
from tqdm import tqdm
import os

SMILES_DIR = "SMILES/data/smiles"
FP_DIR = "SMILES/data/fingerprints"


def get_all_files_in_dir(directory):
    files = []
    for entry in os.listdir(directory):
        entry_path = os.path.join(directory, entry)
        if os.path.isfile(entry_path):
            files.append(entry_path)
    return files

def fingerprint(SMILES, fp_type, fp_size=2048):
    """
    Génère l'empreinte moléculaire (fingerprint) d'une molécule à partir de son SMILES.

    Arguments
    ----------
    SMILES1 : str
        Représentation SMILES
    fp_type : str
        Type de fingerprint à générer. Les options possibles sont :
        - "morgan" : Morgan fingerprints (basé sur des sous-structures circulaires).
        - "ap" : Atom Pair fingerprints (paires d'atomes et leurs distances).
        - "rdkit" : RDKit fingerprints (basé sur des chemins d'atomes).
        - "maccs" : MACCS keys (basé sur 166 sous-structures prédéfinies).
    fp_size : int, optional
        Taille de l'empreinte (nombre total de bits). La valeur par défaut est 2048.

    Retourne:
    -------
    ExplicitBitVect
        Une empreinte moléculaire correspondant à la molécule donnée.

    Raises
    ------
    ValueError
        Si le type de fingerprint ('fp_type') n'est pas valide.
    """

    # Construction d'un objet Molecule à partir d'un SMILES
    mol = Chem.MolFromSmiles(SMILES)

    # Génération des Morgan fingerprints
    if fp_type == "morgan":
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)
        fp = generator.GetFingerprint(mol)
    #Génération des Atom Pair Fingerprints
    elif fp_type == "ap":
        generator = rdFingerprintGenerator.GetAtomPairGenerator(minDistance=1, maxDistance=30, fpSize=fp_size)
        fp = generator.GetFingerprint(mol)
    # Génération des Rdkit fingerprints
    elif fp_type == "rdkit":
        fp = Chem.RDKFingerprint(mol, maxPath=7, fpSize=fp_size)
    # Génération des MACCS fingerprints
    elif fp_type == "maccs":
        fp = MACCSkeys.GenMACCSKeys(mol)
    else:
        raise ValueError("Le type de fingerprint n'est pas valide.")

    arr = np.zeros((1,))
    return DataStructs.ConvertToNumpyArray(fp, arr)

def fingerprint_morgan(SMILES):
    # Construction d'un objet Molecule à partir d'un SMILES
    mol = Chem.MolFromSmiles(SMILES)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = generator.GetFingerprint(mol)
    #return DataStructs.ConvertToNumpyArray(fp, arr)
    return fp.ToList()


def dist_sim(fp1, fp2, dist_type):
    """
    Calcule la distance et la similarité entre deux fingerprints.

    Arguments
    ----------
    fp1 : ExplicitBitVect
        Le premier fingerprint
    fp2 : ExplicitBitVect
        Le second fingerprint
    dist_type : str
        Type de mesure de distance/similarité à utiliser. Les options disponibles sont :
        - "cos" : Distance et similarité cosinus
        - "jacc" : Distance et similarité de Jaccard (Tanimoto)

    Retourne:
    -------
    distance : float
        La distance calculée entre les deux fingerprints.
    similarity : float
        La similarité calculée entre les deux fingerprints.

    Raises
    ------
    ValueError
        Si `dist_type` n'est pas valide.
    """

    # Calcul de la distance & similarité cosinus entre fingerprints
    if dist_type == "cos":
        # Conversion des empreintes en vecteurs numpy
        arr1 = np.zeros((1,))
        arr2 = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp1, arr1)
        DataStructs.ConvertToNumpyArray(fp2, arr2)

        distance = cosine_distances([arr1], [arr2])[0][0] 
        similarity = 1 - distance

    # Calcul de la distance & similarité de Jaccard (Tanimoto)
    elif dist_type == "jacc":
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        distance = 1 - similarity

    else:
        raise ValueError("Le type de distance n'est pas valide.")
    
    return distance, similarity

def write_all_fingerprints(filename, smiles_name):
    smiles = readfile_without_cn(smiles_name)
    with open(filename, "x") as file:
        for smile in tqdm(smiles):
            fp = fingerprint_morgan(smile)
            for val in fp[:-1]:
                file.write(f"{val}")
            file.write(f"{fp[-1]}")
            file.write("\n")

if __name__ == "__main__":
    SMILES_PATHS = get_all_files_in_dir(SMILES_DIR)
    for smilesPath in SMILES_PATHS:
        write_all_fingerprints(f"{FP_DIR}/{smilesPath[19:-11]}_fp.txt", smilesPath)
def morgan_fingerprint(SMILES, fp_size=2048):
    """
    Génère un fingerprint de type Morgan à partir d'une représentation SMILES.

    Arguments
    ----------
    SMILES : str
        Représentation SMILES de la molécule.
    fp_size : int, optional
        Taille du fingerprint généré (par défaut 2048 bits).

    Retourne
    -------
    rdkit.DataStructs.cDataStructs.ExplicitBitVect
        Un vecteur binaire représentant l'empreinte moléculaire de la molécule.
    """
    
    # Construction d'un objet Molecule à partir d'un SMILES
    mol = Chem.MolFromSmiles(SMILES)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)
    fp = generator.GetFingerprint(mol)

    return fp


def fp_matrix_distance(liste_smiles):
    """
    Calcule la matrice des distances cosinus entre fingerprints
    à partir d'une liste de SMILES (str)

    Arguments
    ----------
    liste_smiles : liste de str
        Liste de chaînes SMILES représentant les molécules.

    Retourne
    -------
    np.ndarray
        Une matrice 2D (numpy array) contenant les distances cosinus entre chaque paire de molécules.
        La valeur en (i, j) représente la distance entre la molécule i et la molécule j.
    """
    
    # Génère les fingerprints pour tous les SMILES
    fingerprints = [morgan_fingerprint(smile) for smile in liste_smiles]

    # Convertion en array numpy 2D
    numpy_smiles = np.vstack(fingerprints)

    distance_matrix = cosine_distances(numpy_smiles)

    return distance_matrix