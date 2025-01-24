from fingerprints import *
from smiles_utils import *
import time
import numpy as np

smiles = readfile_without_cn("SMILES/smiles_without_cn.txt")
SMILE1 = smiles[0]
SMILE2 = smiles[1]
FP1 = fingerprint_morgan(SMILE1)
FP2 = fingerprint_morgan(SMILE2)

def custom_jacc(fp1, fp2):
    return len(set(fp1) & set(fp2)) / len(set(fp1) | set(fp2))

def test_jaccard(smile1, smile2):
    for _ in range(10000):
        similarity_jaccard(smile1, smile2)

def test_fingerprint(smile):
    for _ in range(10000):
        fingerprint_morgan(smile)

def test_custom_jaccard(fp1, fp2):
    for _ in range(10000):
        custom_jacc(fp1, fp2)

def main():
    start_jacc = time.time()
    test_custom_jaccard(FP1, FP2)
    stop_jacc = time.time()

    start_fp = time.time()
    test_fingerprint(SMILE1)
    stop_fp = time.time()

    print(f"Temps d'exécution de la similarité de Jaccard: {stop_jacc - start_jacc}")
    print(f"Temps d'exécution de la génération de fingerprint: {stop_fp - start_fp}")

def jaccard_binary(x,y):
    """A function for finding the similarity between two binary vectors"""
    intersection = np.logical_and(x, y)
    union = np.logical_or(x, y)
    similarity = intersection.sum() / float(union.sum())
    return similarity

# Define some binary vectors
x = [0,0,0,0,0,0,0,0,1]
y = [0,0,1,0,0,0,0,0,1]
z = [1,1,0,0,0,1,0,0,0]

# Find similarity among the vectors
simxy = jaccard_binary(x,y)
simxz = jaccard_binary(x,z)
simyz = jaccard_binary(y,z)

print(' Similarity between x and y is', simxy, '\n Similarity between x and z is ', simxz, '\n Similarity between x and z is ', simyz)