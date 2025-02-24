import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Charger le jeu de données Iris
data = load_iris()
X, y = data.data, data.target

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Entraîner un modèle
model = RandomForestClassifier()
model.fit(X_train, y_train)

model.target_names = data.target_names

# Sauvegarder le modèle entraîné dans un fichier .pkl
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Modèle entraîné et sauvegardé sous model.pkl")