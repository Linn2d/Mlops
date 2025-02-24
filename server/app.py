import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# Charger le modèle sauvegardé
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


# Définir la structure des données attendues
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Créer l'application FastAPI
app = FastAPI()

@app.post("/predict")
def predict(features: IrisFeatures):
    # Convertir les données en format attendu par le modèle
    input_data = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]]
    # Faire une prédiction
    prediction = model.predict(input_data)
    predicted_class_index = prediction[0]
    # Récupérer le nom de la classe correspondante
    predicted_class_name = model.target_names[predicted_class_index]
    return {"predicted_class": predicted_class_name}