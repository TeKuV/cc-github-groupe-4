# 📊 Projet de Prédiction du Prix des Diamants avec PyTorch  
**Groupe 4** : TEUGA Ulirch, HiroSHI, TATSA Colince  
*Date du projet* : 14/03/2025 

---

## 📖 Table des Matières  
1. [Introduction](#-introduction)  
2. [Aperçu du Dataset](#-aperçu-du-dataset)  
3. [Structure du Projet](#-structure-du-projet)  
4. [Installation et Configuration](#-installation-et-configuration)  
5. [Utilisation du Projet](#-utilisation-du-projet)  
6. [Méthodologie](#-méthodologie)  
7. [Résultats et Métriques](#-résultats-et-métriques)  
8. [Contribution des Membres](#-contribution-des-membres)  
9. [Améliorations Futures](#-améliorations-futures)  
10. [Licence](#-licence)  
11. [Références](#-références)  
12. [FAQ](#-faq)  

---

## 🌟 Introduction  
Ce projet vise à prédire le prix des diamants à partir de caractéristiques physiques (carat, couleur, pureté, etc.) en utilisant **PyTorch**.  
- **Objectif** : Développer un modèle de Deep Learning capable de généraliser sur des données structurées.  
- **Dataset** : `diamonds.csv` (contient 53,940 entrées avec 10 caractéristiques).  
- **Public cible** : Joailliers, collectionneurs, ou plateformes e-commerce de luxe.  

---

## 🔍 Aperçu du Dataset  
### 📂 Source  
- **Dataset** : [Diamonds Dataset](https://www.kaggle.com/datasets/shivam2503/diamonds) (Kaggle).  
- **Caractéristiques clés** :  
  - **Numériques** : `carat`, `depth`, `table`, `x`, `y`, `z`, `price` (target).  
  - **Catégorielles** : `cut`, `color`, `clarity`.  

### 🛠 Pré-traitement  
- Gestion des valeurs manquantes : Aucune dans ce dataset.  
- Encodage des variables catégorielles : **One-Hot Encoding** pour `cut`, `color`, `clarity`.  
- Normalisation : Standardisation des features numériques avec `StandardScaler`.  

---

## 🗂 Structure du Projet  
```plaintext
diamond-price-prediction/  
├── data/  
│   ├── diamonds.csv              # Dataset original  
│   └── processed/                # Données pré-traitées  
├── models/  
│   ├── model.py                  # Architecture du modèle  
│   └── trained_model.pth         # Modèle entraîné  
├── notebooks/  
│   ├── EDA.ipynb                 # Analyse exploratoire  
│   └── Training.ipynb            # Entraînement du modèle  
├── src/  
│   ├── preprocess.py             # Script de pré-traitement  
│   ├── train.py                  # Script d'entraînement  
│   └── evaluate.py               # Script d'évaluation  
├── requirements.txt              # Dépendances  
└── README.md                     # Ce fichier  
