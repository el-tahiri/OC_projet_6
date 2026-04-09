# 🏙️ Prédiction de la consommation énergétique des bâtiments – Seattle

## 📌 Contexte

Dans le cadre de la stratégie de la ville de Seattle visant à devenir **neutre en carbone d’ici 2050**, ce projet consiste à analyser et modéliser la consommation énergétique des bâtiments non résidentiels.

Les données utilisées proviennent de relevés réalisés en **2016** par la ville.

L’objectif est de construire un modèle capable de **prédire la consommation d’énergie** à partir de caractéristiques structurelles des bâtiments.

---

## 🎯 Objectifs

* Réaliser une **analyse exploratoire (EDA)** du dataset
* Effectuer du **feature engineering**
* Tester plusieurs **modèles de machine learning**
* Optimiser le modèle sélectionné
* Identifier les **variables les plus influentes**
* Déployer le modèle via une **API avec BentoML**
* Conteneuriser avec **Docker**
* Déployer sur **Google Cloud Run**

---

## 📊 Données

* Source : Ville de Seattle
* Fichier : `2016_Building_Energy_Benchmarking.csv`
* Localisation : `./data/`

Les données incluent :

* caractéristiques des bâtiments (surface, usage, année de construction…)
* consommation énergétique
* émissions de CO₂

---

## 🧪 Méthodologie

### 1. Analyse exploratoire

* Nettoyage des données
* Analyse des distributions
* Détection des valeurs aberrantes

### 2. Feature Engineering

* Création de nouvelles variables :

  * `BuildingAge`
  * `AreaPerFloor`
  * `NbUses`
  * indicateurs d’énergie (`HasElectricity`, `HasGas`)
* Encodage des variables catégorielles

### 3. Modélisation

* Modèles testés :

  * Régression linéaire
  * Random Forest
  * SVR
* Évaluation avec :

  * R²
  * MAE
  * RMSE

### 4. Optimisation

* GridSearchCV
* Sélection du meilleur modèle

### 5. Interprétation

* Analyse des **feature importances**

---

## 🚀 API – Déploiement du modèle

Une API a été développée avec **BentoML** permettant de prédire la consommation d’un bâtiment.

### Endpoint

```
POST /predict
```

### Exemple de requête

```json
{
  "ENERGYSTARScore": 75,
  "PropertyGFABuilding(s)": 120000,
  "LargestPropertyUseTypeGFA": 90000,
  "ZipCode": 98101,
  "NumberofFloors": 12,
  "YearBuilt": 1998,
  "PrimaryPropertyType": "Hotel",
  "LargestPropertyUseType": "Hotel"
}
```

### Réponse

```json
{
  "prediction": 123.45
}
```

---

## 🐳 Docker

### Build

```bash
docker build -t api_energy .
```

### Run

```bash
docker run -p 3000:3000 api_energy
```

---

## ☁️ Déploiement Cloud

API déployée sur **Google Cloud Run** :

🔗 https://api-energy-610310238285.europe-west1.run.app/

---

## 📁 Structure du projet

```
Seattle_projet/
│
├── data/
│   └── dataset CSV
│
├── notebook/
│   └── analyse + modélisation
│
├── src/
│   ├── best_model.py
│   └── service.py
│
├── Dockerfile
├── requirements.txt
└── test_api.http
```

---

## ⚙️ Installation locale

```bash
git clone <repo>
cd Seattle_projet
pip install -r requirements.txt
python src/best_model.py
bentoml serve src/service:svc
```

---

## 🧠 Technologies utilisées

* Python
* Pandas / NumPy
* Scikit-learn
* BentoML
* Docker
* Google Cloud Run

---

## 👨‍💻 Auteur

Projet réalisé dans le cadre d’une formation en Data Engineering / Data Science.

---

## ⚠️ Remarque

Ce projet a pour but pédagogique de démontrer :

* la construction d’un modèle ML
* son exposition via API
* son déploiement en production
