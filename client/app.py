import requests
import streamlit as st

# URL de l'API
API_URL = "http://server:8000/predict"

st.title("Prédiction des classes Iris")
st.write("Entrez les caractéristiques pour obtenir une prédiction.")

# Entrées utilisateur
sepal_length = st.number_input("Longueur du sépale")
sepal_width = st.number_input("Largeur du sépale")
petal_length = st.number_input("Longueur du pétale")
petal_width = st.number_input("Largeur du pétale")

# Bouton de prédiction
if st.button("Prédire"):
    # Envoyer les données au serveur
    payload = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }
    response = requests.post(API_URL, json=payload)
    if response.status_code == 200:
        prediction = response.json()["predicted_class"]
        st.success(f"La classe prédite est : {prediction}")
    else:
        st.error("Erreur lors de la prédiction.")