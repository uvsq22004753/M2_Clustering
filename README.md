# Clustering de molécules

*~~CHAABANI Ahmed~~*  
*CIESLA Julie*  
*GROSJACQUES Marwane*  
*HOSTI Pauline*  
*RICOUR-DUMAS Lazare*  
*TABET Mohamed Cherif*  

## :mag: Présentation

Ce projet a été réalisé dans le cadre du module _Projet_ du Master AMIS. Le but est d’étudier la similarité des spectres de masse d’une part et des SMILES d’autre part, puis de mesurer si l’on retrouve des points communs dans ces deux représentations.

## :hammer: Installation

Plusieurs bibliothèques Python sont nécessaires :

* **NumPy** permet d'effectuer des calculs mathématiques rapidement :zap:.

* **Pyteomics** permet de lire et écrire des fichiers au format MGF :scroll:.

* **matchms** offre divers outils liés aux spectres de masse et à la génération de fingerprints de SMILES :atom:.

* **scikit-learn** propose des méthodes de clustering :sparkles:.

* **tqdm** permet d'afficher des barres de progression dans les boucles ou autres processus itératifs :signal_strength:.

Toutes ces bibliothèques se trouvent dans le fichier `requirements.txt` et peuvent être installées avec la commande suivante :  
```bash
pip install -r requirements.txt
```

## :technologist: Utilisation

Le script `src/cli.py` permet d'exécuter différentes commandes (comme le parsing ou le calcul des clusters). Il s'utilise de la manière suivante :
```bash
python3 cli.py <command> [options]
```

La liste des commandes est disponible avec :
```bash
python3 cli.py --help
```

Le détail et les options de chaque commande sont disponibles avec :
```bash
python3 cli.py <command> --help
```

## :link: Les données

Nous avons utilisé le fichier [ALL_GNPS_cleaned.mgf](https://zenodo.org/records/11193898) lors de ce projet. Il comporte 542 277 molécules pour une taille de 1,7 Go.