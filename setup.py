from setuptools import setup, find_packages

setup(
    name="spectra_analyser",
    version="0.1.0",
    title="Clustering de molécules",
    description="Analyse et traitement des spectres de masse et calcul de similarité des SMILES.",
    long_description=(
        "🔍 Présentation\n"
        "Ce projet a été réalisé dans le cadre du module Projet du Master AMIS. "
        "Le but est d’étudier la similarité de spectres de masse d’une part et des SMILES d’autre part, "
        "puis de mesurer si l’on retrouve des points communs dans ces deux représentations."
    ),
    author="CIESLA Julie, GROSJACQUES Marwane, HOSTI Pauline, RICOUR-DUMAS Lazare, TABET Mohamed Cherif",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    py_modules=["cli", "config"],
    install_requires=[
        "numpy",
        "pyteomics",
        "matchms",
        "scikit-learn",
        "tqdm",
        "hdbscan",
    ],
    entry_points={
        "console_scripts": [
            "spectra_analyser = cli:main",
        ],
    },
)
