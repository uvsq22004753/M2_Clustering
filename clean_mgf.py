def clean(mgf_path, path_out):
    with open(mgf_path, 'r') as file:
        smiles = []
        for line in file.readlines():
            if line[0:7] == "SMILES=":
                smiles.append(line[7:])

    with open(path_out, 'x') as out:
        out.writelines(smiles)


if __name__ == "__main__":
    clean("adducts/[M-3H2O+H]1+.mgf", "SMILES/data/smiles/[M-3H2O+H]1+_smiles.txt")
    clean("adducts/[M+Ca]2+.mgf", "SMILES/data/smiles/[M+Ca]2+_smiles.txt")
    clean("adducts/[2M+Ca]2+.mgf", "SMILES/data/smiles/[2M+Ca]2+_smiles.txt")
    clean("adducts/[2M+H]1+.mgf", "SMILES/data/smiles/[2M+H]1+_smiles.txt")
    clean("adducts/[2M+NH3+H]1+.mgf", "SMILES/data/smiles/[2M+NH3+H]1+_smiles.txt")
    clean("adducts/[M-H2O+H]1+.mgf", "SMILES/data/smiles/[M-H2O+H]1+_smiles.txt")
    clean("adducts/[M+2H]2+.mgf", "SMILES/data/smiles/[M+2H]2+_smiles.txt") 