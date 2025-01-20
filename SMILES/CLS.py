def CLS(smile1, smile2):
    """
    Cette fonction calcule la longueur de la plus longue sous-séquence commune (CLS) entre deux chaînes SMILES.
    Arguments:
    ----------
        smile1: str
            La première chaîne SMILES.
        smile2: str
            La deuxième chaîne SMILES.
    Retourne:
    ----------
        int: La longueur de la plus longue sous-séquence commune entre les deux chaînes SMILES.
    """
    m, n = len(smile1), len(smile2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if smile1[i - 1] == smile2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]