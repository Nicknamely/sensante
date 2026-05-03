# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd  # Ajout de pandas pour eviter les warnings de noms de colonnes


# --- 1. Schemas Pydantic ---
class PatientInput(BaseModel):
    """Donnees d'entree : Ajout de frissons et nausee pour atteindre 10 features."""

    age: int = Field(..., ge=0, le=120)
    sexe: str = Field(...)  # 'M' ou 'F'
    temperature: float = Field(..., ge=35.0, le=42.0)
    tension_sys: int = Field(..., ge=60, le=250)
    toux: bool = Field(...)
    fatigue: bool = Field(...)
    maux_tete: bool = Field(...)
    frissons: bool = Field(...)  # Nouvelle feature du Lab 2
    nausee: bool = Field(...)  # Nouvelle feature du Lab 2
    region: str = Field(...)


class DiagnosticOutput(BaseModel):
    diagnostic: str
    probabilite: float
    confiance: str
    message: str


# --- 2. Application ---
app = FastAPI(title="SenSante API", version="0.2.1")

# --- 3. Chargement ---
print("Chargement du modele et des encodeurs...")
model = joblib.load("models/model.pkl")
le_sexe = joblib.load("models/encoder_sexe.pkl")
le_region = joblib.load("models/encoder_region.pkl")
feature_cols = joblib.load(
    "models/feature_cols.pkl"
)  # Contient les 10 noms de colonnes


# --- 4. Routes ---
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):
    # 1. Encodage
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
        region_enc = le_region.transform([patient.region])[0]
    except ValueError as e:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Valeur incorrecte : {str(e)}",
        )

    # 2. Preparation des donnees (DataFrame pour garder les noms de colonnes)
    input_df = pd.DataFrame(
        [
            [
                patient.age,
                sexe_enc,
                patient.temperature,
                patient.tension_sys,
                int(patient.toux),
                int(patient.fatigue),
                int(patient.maux_tete),
                int(patient.frissons),
                int(patient.nausee),
                region_enc,
            ]
        ],
        columns=feature_cols,
    )

    # 3. Prediction
    diagnostic = model.predict(input_df)[0]
    proba_max = float(model.predict_proba(input_df).max())

    confiance = (
        "haute" if proba_max >= 0.7 else "moyenne" if proba_max >= 0.4 else "faible"
    )

    messages = {
        "palu": "Suspicion de paludisme. Consultez un medecin.",
        "grippe": "Suspicion de grippe. Repos conseille.",
        "typh": "Suspicion de typhoide. Consultation necessaire.",
        "sain": "Pas de pathologie detectee.",
    }

    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(diagnostic, "Consultez un medecin."),
    )
