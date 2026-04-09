# ============================================
# 1. IMPORTS
# ============================================
import pandas as pd
import numpy as np
import bentoml

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor


# ============================================
# 2. CHARGEMENT DES DONNÉES
# ============================================
df = pd.read_csv("2016_Building_Energy_Benchmarking.csv")


# ============================================
# 3. SUPPRESSION COLONNES TROP VIDES
# ============================================
cols_unit = df.loc[:, df.count() <= int(len(df) / 2)]
df = df.drop(columns=cols_unit.columns).copy()


# ============================================
# 4. ZIPCODE CLEANING
# ============================================
zipmq = df[df["ZipCode"].isna()]

for idx, row in zipmq.iterrows():
    match = df[
        (df["Latitude"] == row["Latitude"]) &
        (df["Longitude"] == row["Longitude"]) &
        (df["ZipCode"].notna())
    ]
    if not match.empty:
        df.at[idx, "ZipCode"] = match.iloc[0]["ZipCode"]

df = df[df["ZipCode"].notna()].copy()


# ============================================
# 5. IMPUTATION
# ============================================
df["ENERGYSTARScore"] = df["ENERGYSTARScore"].fillna(df["ENERGYSTARScore"].median())
df = df[df["NumberofBuildings"].notna()]
df = df[df["TotalGHGEmissions"].notna()]


# ============================================
# 6. FEATURE ENGINEERING
# ============================================
df["BuildingAge"] = 2016 - df["YearBuilt"]

df["AreaPerFloor"] = (
    df["PropertyGFABuilding(s)"] /
    df["NumberofFloors"].replace(0, 1)
)

df["NbUses"] = df["ListOfAllPropertyUseTypes"].fillna("").apply(
    lambda x: len(x.split(",")) if x != "" else 0
)

df["HasElectricity"] = (df["Electricity(kWh)"] > 0).astype(int)
df["HasGas"] = (df["NaturalGas(therms)"] > 0).astype(int)


# ============================================
# 7. DROP COLONNES INUTILES
# ============================================
col_sup = [
    "OSEBuildingID", "PropertyName", "Address", "City", "State",
    "TaxParcelIdentificationNumber",
    "DataYear", "DefaultData", "ComplianceStatus",
    "YearBuilt", "PropertyGFATotal", "PropertyGFAParking",
    "ListOfAllPropertyUseTypes",
    "SiteEUIWN(kBtu/sf)", "SourceEUI(kBtu/sf)", "SourceEUIWN(kBtu/sf)",
    "SiteEnergyUse(kBtu)", "SiteEnergyUseWN(kBtu)",
    "SteamUse(kBtu)", "Electricity(kWh)", "Electricity(kBtu)",
    "NaturalGas(therms)", "NaturalGas(kBtu)",
    "TotalGHGEmissions", "GHGEmissionsIntensity"
]

df = df.drop(columns=col_sup, errors="ignore")


# ============================================
# 8. X / y
# ============================================
y = df["SiteEUI(kBtu/sf)"]
X = df.drop(columns=["SiteEUI(kBtu/sf)"])


# ============================================
# 9. ENCODAGE
# ============================================
X = pd.get_dummies(X, drop_first=True)

# bool → int
bool_cols = X.select_dtypes(include="bool").columns
X[bool_cols] = X[bool_cols].astype(int)


# ============================================
# 10. NETTOYAGE FINAL
# ============================================
y = y.fillna(y.median())
X = X.fillna(X.median())


# ============================================
# 11. SPLIT
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ============================================
# 12. GRID SEARCH (MODEL FINAL)
# ============================================
param_grid = {
    "n_estimators": [100, 500],
    "max_depth": [10, 100],
    "min_samples_split": [2, 50]
}

grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_


# ============================================
# 13. FEATURE IMPORTANCE
# ============================================
importances = best_model.feature_importances_

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

top_features = feature_importance.head(15)["feature"].tolist()


# ============================================
# 14. SAVE MODEL (BENTOML)
# ============================================
bentoml.sklearn.save_model(
    "energy_model",
    best_model,
    custom_objects={
        "columns": X.columns.tolist(),
        "top_features": top_features,
        "feature_relations": {
            "BuildingAge": {
                "inputs": ["YearBuilt"],
                "formula": "2016 - YearBuilt"
            },
            "AreaPerFloor": {
                "inputs": ["PropertyGFABuilding(s)", "NumberofFloors"],
                "formula": "PropertyGFABuilding(s) / NumberofFloors"
            }
        }
    }
)

print("✅ Modèle sauvegardé avec succès dans BentoML")