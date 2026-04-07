import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

# ==========================================
# 1. GENERATE SYNTHETIC GENETIC DATA
# ==========================================
np.random.seed(42)
n_patients = 1000  # Increased for better training

data = {
    'Gene_A': np.random.choice([0, 1], n_patients, p=[0.90, 0.10]),
    'Gene_B': np.random.choice([0, 1], n_patients, p=[0.95, 0.05]),
    'Gene_C': np.random.choice([0, 1], n_patients, p=[0.85, 0.15]),
    'Protein_Level': np.random.normal(100, 15, n_patients),
    'Age': np.random.randint(1, 18, n_patients)
}
df = pd.DataFrame(data)

# Logic: Rare Disease occurs only if Gene_A AND Gene_B are mutated (1)
df['Disease_Status'] = ((df['Gene_A'] == 1) & (df['Gene_B'] == 1)).astype(int)

# ==========================================
# 2. TRAIN THE GRADIENT BOOSTER
# ==========================================
X = df.drop('Disease_Status', axis=1)
y = df['Disease_Status']

# Stratify ensures the rare cases are split evenly between train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

model = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1)
model.fit(X_train, y_train)

# ==========================================
# 3. GENETIC DIAGNOSTIC INTERFACE
# ==========================================
def screen_patient(name, gene_a, gene_b, gene_c, protein, age):
    sample = [[gene_a, gene_b, gene_c, protein, age]]
    # We use predict_proba to see the "Certainty" of the model
    prob = model.predict_proba(sample)[0][1]
    
    print(f"\n--- Genetic Report: {name} (Age {age}) ---")
    print(f"Markers: A:{gene_a}, B:{gene_b}, C:{gene_c} | Protein: {protein}")
    print(f"Risk Probability: {prob:.2%}")
    
    if prob > 0.5:
        print("🚨 STATUS: HIGH RISK. Rare Variant Detected.")
    else:
        print("✅ STATUS: NEGATIVE. No rare hereditary markers found.")

# ==========================================
# 4. FINAL PORTFOLIO TEST CASES
# ==========================================
print(f"System Readiness: {model.score(X_test, y_test):.2%} Accuracy")

# CASE 1: The "High Risk" Patient (Has both mutations)
screen_patient("Patient_001", 1, 1, 0, 85, 5)

# CASE 2: The "Healthy" Patient (No mutations)
screen_patient("Patient_002", 0, 0, 1, 105, 12)

# CASE 3: The "Carrier" (Only has one mutation, should be negative)
screen_patient("Patient_003", 1, 0, 1, 98, 8)
