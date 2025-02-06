from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np

def gen_lingos(smile: str, n: int = 3) -> np.ndarray:
    """
    Génère les n-grammes (lingos) de la chaîne SMILES.
    """
    # Exemple: pour n=3, on génère tous les trigrammes.
    return np.array([smile[i:i+n] for i in range(len(smile) - n + 1)])

def morgan_fingerprint(smile: str, fp_size: int = 2048):
    """
    Génère un fingerprint de type Morgan à partir d'un SMILES.
    Retourne un objet rdkit.DataStructs.cDataStructs.ExplicitBitVect.
    """
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"Impossible de convertir le SMILES '{smile}' en molécule RDKit.")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)
    return generator.GetFingerprint(mol)
