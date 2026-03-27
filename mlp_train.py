import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import joblib
import os

MODEL_PATH = "models/mlp_model.pkl"

def train_mlp():
    # Load dataset
    df = pd.read_csv("data/thyroidDFF.csv")
    df.columns = df.columns.str.strip()

    # Keep only required columns
    df = df[['TSH', 'T3', 'T4', 'target']]

    # Remove rows where target is missing
    df = df[df['target'].notna()]

    # ================= TARGET MAPPING =================
    def map_target(val):
        val = str(val).strip().upper()

        if val == "-":
            return "Normal"
        elif val in ["S", "F", "I"]:  
            return "Hyperthyroidism"
        elif val in ["O", "H"]:  
            return "Hypothyroidism"
        else:
            return None

    df['Diagnosis'] = df['target'].apply(map_target)

    # Remove unknown labels
    df = df[df['Diagnosis'].notna()]

    # ================= FEATURES =================
    X = df[['TSH', 'T3', 'T4']].values
    y = df['Diagnosis'].values

    # ================= HANDLE NaN =================
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    # ================= ENCODE LABELS =================
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # ================= TRAIN MLP =================
    model = MLPClassifier(hidden_layer_sizes=(20, 20), max_iter=600, random_state=42)
    model.fit(X, y_encoded)

    # ================= SAVE =================
    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "model": model,
        "label_encoder": le,
        "imputer": imputer
    }, MODEL_PATH)

    print("✅ MLP trained using REAL dataset (no rules used)")

if __name__ == "__main__":
    train_mlp()