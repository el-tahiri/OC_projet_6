from __future__ import annotations

import numpy as np
import pandas as pd
import bentoml
from bentoml.models import BentoModel
from pydantic import BaseModel, Field, field_validator


MODEL_TAG = "energy_model:latest"
ENERGY_YEAR = 2016


class BuildingInput(BaseModel):
    energy_star_score: float = Field(alias="ENERGYSTARScore", ge=1, le=100)
    property_gfa_building: float = Field(alias="PropertyGFABuilding(s)", gt=0)
    largest_property_use_type_gfa: float = Field(alias="LargestPropertyUseTypeGFA", gt=0)
    zip_code: int = Field(alias="ZipCode", ge=98000, le=98300)
    number_of_floors: int = Field(alias="NumberofFloors", ge=1, le=150)
    year_built: int = Field(alias="YearBuilt", ge=1800, le=ENERGY_YEAR)
    primary_property_type: str = Field(alias="PrimaryPropertyType", min_length=1)
    largest_property_use_type: str = Field(alias="LargestPropertyUseType", min_length=1)

    model_config = {
        "populate_by_name": True,
        "extra": "forbid",
        "json_schema_extra": {
            "example": {
                "ENERGYSTARScore": 75,
                "PropertyGFABuilding(s)": 120000,
                "LargestPropertyUseTypeGFA": 90000,
                "ZipCode": 98101,
                "NumberofFloors": 12,
                "YearBuilt": 1998,
                "PrimaryPropertyType": "Hotel",
                "LargestPropertyUseType": "Hotel",
            }
        },
    }

    @field_validator("primary_property_type", "largest_property_use_type")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("La valeur ne peut pas être vide.")
        return value


class PredictionOutput(BaseModel):
    prediction: float


@bentoml.service(name="energy_prediction_service")
class EnergyPredictionService:
    model_ref = BentoModel(MODEL_TAG)

    def __init__(self) -> None:
        self.bento_model = self.model_ref
        self.model = bentoml.sklearn.load_model(self.bento_model)
        self.model_columns = self.bento_model.custom_objects.get("columns", [])
        self.top_features = self.bento_model.custom_objects.get("top_features", [])
        self.feature_relations = self.bento_model.custom_objects.get("feature_relations", {})

    @bentoml.api(route="/predict")
    def predict(self, payload: BuildingInput) -> PredictionOutput:
        input_row = pd.DataFrame(
            np.zeros((1, len(self.model_columns)), dtype=float),
            columns=self.model_columns,
        )

        numeric_values = {
            "ENERGYSTARScore": payload.energy_star_score,
            "PropertyGFABuilding(s)": payload.property_gfa_building,
            "LargestPropertyUseTypeGFA": payload.largest_property_use_type_gfa,
            "ZipCode": float(payload.zip_code),
            "NumberofFloors": float(payload.number_of_floors),
        }

        for col_name, value in numeric_values.items():
            if col_name in input_row.columns:
                input_row.at[0, col_name] = value

        if "BuildingAge" in input_row.columns:
            input_row.at[0, "BuildingAge"] = ENERGY_YEAR - payload.year_built

        if "AreaPerFloor" in input_row.columns:
            floors = payload.number_of_floors if payload.number_of_floors != 0 else 1
            input_row.at[0, "AreaPerFloor"] = payload.property_gfa_building / floors

        primary_col = f"PrimaryPropertyType_{payload.primary_property_type}"
        if primary_col in input_row.columns:
            input_row.at[0, primary_col] = 1.0

        largest_col = f"LargestPropertyUseType_{payload.largest_property_use_type}"
        if largest_col in input_row.columns:
            input_row.at[0, largest_col] = 1.0

        prediction = float(self.model.predict(input_row)[0])
        return PredictionOutput(prediction=prediction)


svc = EnergyPredictionService