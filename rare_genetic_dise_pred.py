import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# ==========================================================
# 1. GENERATE SYNTHETIC GENETIC DATA (Precision Medicine)
# ==========================================================
np.random.seed(42)
n_patients = 1000 

feature_names = ['Gene_A', 'Gene_B', 'Gene_C', 'Protein_Level', 'Age']

data = {
    'Gene_A': np.random.choice([0, 1], n_patients, p=[0.90, 0.10]),
    'Gene_B': np.random.choice([0, 1], n_patients, p=[0.95, 0.05]),
    'Gene_C': np.random.choice([0, 1], n_patients, p=[0.85, 0.15]),
    'Protein_Level': np.random.normal(100, 15, n_patients),
    'Age': np.random.randint(1, 18, n_patients)
}
df = pd.DataFrame(data)

# Logic: Epistatic Interaction (Disease only if Gene_A AND Gene_B == 1)
df['Disease_Status'] = ((df['Gene_A'] == 1) & (df['Gene_B'] == 1)).astype(int)

# ==========================================================
# 2. TRAIN THE GRADIENT BOOSTER (Classification Engine)
# ==========================================================
X = df[feature_names]
y = df['Disease_Status']

# Stratify ensures the rare cases are represented in both sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

model = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1)
model.fit(X_train, y_train)

# ==========================================================
# 3. GENETIC DIAGNOSTIC INTERFACE (Feature Aligned)
# ==========================================================
def screen_patient(name, gene_a, gene_b, gene_c, protein, age):
    # Fixed: Passing a DataFrame with names to stop the pink warnings
    sample_df = pd.DataFrame([[gene_a, gene_b, gene_c, protein, age]], columns=feature_names)
    
    prob = model.predict_proba(sample_df)[0][1]
    
    print(f"\n--- Genetic Report: {name} (Age {age}) ---")
    print(f"Markers: A:{gene_a}, B:{gene_b}, C:{gene_c} | Protein: {protein:.1f}")
    print(f"Risk Probability: {prob:.2%}")
    
    if prob > 0.5:
        print("🚨 STATUS: HIGH RISK. Rare Variant Detected.")
    else:
        print("✅ STATUS: NEGATIVE. No rare hereditary markers found.")

# ==========================================================
# 4. PERFORMANCE EVALUATION
# ==========================================================
print("\n" + "="*40)
print(f"System Accuracy: {model.score(X_test, y_test):.2%}")
print("="*40)

y_pred = model.predict(X_test)
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Test Cases
screen_patient("Patient_001 (Sick)", 1, 1, 0, 85.5, 5)
screen_patient("Patient_003 (Carrier)", 1, 0, 1, 98.2, 8)

# ==========================================================
# 5. MATPLOTLIB VISUALIZATION (No Seaborn)
# ==========================================================
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

# Setup labels
classes = ['Healthy', 'Disease']
tick_marks = np.arange(len(classes))
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)

# Annotate the numbers inside the squares
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

ax.set_title("Diagnostic Performance (Confusion Matrix)")
ax.set_ylabel('Actual Status')
ax.set_xlabel('Predicted Status')
plt.tight_layout()
plt.show()
: