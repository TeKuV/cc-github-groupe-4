
import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scripts.DiamondModel import DiamondModel

# Charger le modèle sauvegardé
MODEL_PATH = "model_final.pth"

model = DiamondModel(9)


# Charger le modèle complet
model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.eval()

# Définir les noms des colonnes d'entrée
feature_columns = ["carat", "depth", "table", "price", "x", "y", "z", "Color", "Clarity"]

# Définir les classes de sortie
class_labels = ['Fair', 'Good', 'Ideal', 'Premium', 'Very Good']

st.title("Prédiction de la qualité du diamant")
st.write("Entrez les caractéristiques du diamant pour prédire sa qualité.")

# Interface utilisateur pour entrer les valeurs des features
st.sidebar.header("Entrée des caractéristiques")
features = {}

col1, col2 = st.sidebar.columns(2)

for i, col in enumerate(feature_columns):
    if i % 2 == 0:
        features[col] = col1.number_input(f"{col}", value=0.0)
    else:
        features[col] = col2.number_input(f"{col}", value=0.0)

if st.sidebar.button("Prédire"):
    input_tensor = torch.tensor([list(features.values())], dtype=torch.float32)
    prediction_index = torch.argmax(model(input_tensor), dim=1).item()
    predicted_class = class_labels[prediction_index]
    st.subheader(f"Ce diamant entre dans la categorie: {predicted_class}")
    st.balloons()
