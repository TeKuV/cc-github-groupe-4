
import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Charger le modèle sauvegardé
MODEL_PATH = "modeleANN.pth"  # Remplace par le vrai chemin

class diamonds_model(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(in_features=9, out_features=15)
        self.layer_2 = nn.Linear(in_features=15, out_features=12)
        self.layer_3 = nn.Linear(in_features=12, out_features=8)
        self.layer_4 = nn.Linear(in_features=8, out_features=5)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.layer_4(x)
        return x
    
model = diamonds_model()

# Définition du modèle
class LinearRegression(nn.Module):
    def __init__(self, input_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

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
