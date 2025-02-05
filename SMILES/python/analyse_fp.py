from fingerprints import fingerprint, dist_sim
from smiles_utils import readfile


def test_best_fp(filename):
    """
    Cette fonction analyse un ensemble de molécules avec leurs représentations SMILES 
    et teste différents types de fingerprints (Morgan, Atom Pair, RDKit et MACCS) en 
    combinant plusieurs mesures de similarité (Jaccard et Cosinus).

    Elle identifie quel type de fingerprint et quelle mesure de distance (similarité) 
    prédit le mieux la similarité entre les SMILES représentant la même molécule.

    Arguments:
    ----------
        filename: str
            Le nom du fichier contenant les données des molécules et leurs représentations SMILES 
                         ("smiles.out").

    Retourne:
    ----------
        dict: Un dictionnaire où chaque clé représente une combinaison de type de fingerprint 
              et de mesure de similarité (par exemple: "morgan-jacc" ou "maccs-cos"), et la 
              valeur associée est le nombre d'occurrences où cette combinaison a produit la 
              meilleure similarité entre des paires de SMILES pour des molécules ayant plusieurs SMILES.
    """

    # Récupération des molécules à partir du fichier
    molecules = readfile(filename)

    # Liste des types de fingerprints et des types de distance
    liste_fp_type = ["morgan", "ap", "rdkit", "maccs"]
    liste_dist_type = ["jacc", "cos"]

    # Initialisation du dictionnaire pour stocker les résultats
    dico = {f"{fp}-{dist}": 0 for fp in liste_fp_type for dist in liste_dist_type}

    # Pour chaque molecule de la liste
    for mol in molecules:
        # Si une molécule a plusieurs SMILES
        if len(mol[1]) >= 2:
            
            SMILES_list = mol[1]  # Liste des SMILES pour cette molécule
            best_sim = -1  # Initialisation de la meilleure similarité
            
            # Comparer les SMILES entre eux
            for i in range(len(SMILES_list)):
                for j in range(i + 1, len(SMILES_list)):
                    SMILES1 = SMILES_list[i]
                    SMILES2 = SMILES_list[j]
                    
                    # Test sur tous les types de fingerprint
                    for fp_type in liste_fp_type:
                        fp1 = fingerprint(SMILES1, fp_type)
                        fp2 = fingerprint(SMILES2, fp_type)
                        
                        # Test sur toutes les distances
                        for dist_type in liste_dist_type:
                            dist, sim = dist_sim(fp1, fp2, dist_type)

                            # Sauvegarde le meilleur résultat
                            # Et l'ajoute au dico
                            if sim >= best_sim:
                                best_sim = sim
                                key = f"{fp_type}-{dist_type}"
                                dico[key] += 1
    return dico

print("Program Running...")
dico = test_best_fp("smiles.out")

for clé, valeur in dico.items():
    print(f"{clé}: {valeur}")