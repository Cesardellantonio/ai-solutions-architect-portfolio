"""
Project 1: ML Pipeline — Iris Classifier
AI Solutions Architect Portfolio | Day 1, Session 2

Setup: pip install -r requirements.txt
Run:   python run_pipeline.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend: saves charts to file
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Save charts next to this script
CHART_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 64)
print("  ML PIPELINE — Project 1: Iris Classifier")
print("=" * 64)

# ─────────────────────────────────────────────────────────────
# CELL 1: Libraries
# ─────────────────────────────────────────────────────────────
print("\n[STEP 1/8] Libraries loaded ✓")
print(f"  pandas {pd.__version__}, numpy {np.__version__}, scikit-learn {__import__('sklearn').__version__}")

# ─────────────────────────────────────────────────────────────
# CELL 2: Load & Explore Data
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 64)
print("[STEP 2/8] Load & Explore the Data")
print("─" * 64)

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print(f"\n  Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Features (inputs):  {iris.feature_names}")
print(f"  Labels (outputs):   {list(iris.target_names)}")
print(f"\n  First 5 rows:")
print(df.head().to_string(index=False))

print(f"\n  Class balance:")
for name in iris.target_names:
    count = (df['species_name'] == name).sum()
    print(f"    {name:15s} → {count} samples")

# ─────────────────────────────────────────────────────────────
# CELL 3: Visualize Feature Distributions
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 64)
print("[STEP 3/8] Visualize Feature Distributions")
print("─" * 64)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Iris Dataset — Feature Distributions by Species', fontsize=14, fontweight='bold')

for i, feature in enumerate(iris.feature_names):
    ax = axes[i // 2, i % 2]
    for species_id, species_name in enumerate(iris.target_names):
        subset = df[df['species'] == species_id]
        ax.hist(subset[feature], alpha=0.6, label=species_name, bins=15)
    ax.set_xlabel(feature, fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.legend()

plt.tight_layout()
chart1_path = os.path.join(CHART_DIR, 'chart_1_feature_distributions.png')
plt.savefig(chart1_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Chart saved → chart_1_feature_distributions.png")

# ─────────────────────────────────────────────────────────────
# CELL 4: Prepare Data (Split + Scale)
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 64)
print("[STEP 4/8] Prepare Data — Split & Normalize")
print("─" * 64)

X = df[iris.feature_names]
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n  Training set: {X_train.shape[0]} samples (80%)")
print(f"  Testing set:  {X_test.shape[0]} samples (20%)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n  Scaling applied:")
print(f"    Before — sepal length range: {X_train['sepal length (cm)'].min():.1f} to {X_train['sepal length (cm)'].max():.1f}")
print(f"    After  — sepal length range: {X_train_scaled[:, 0].min():.2f} to {X_train_scaled[:, 0].max():.2f}")
print(f"\n  fit_transform on TRAIN, transform on TEST (no data leakage)")

# ─────────────────────────────────────────────────────────────
# CELL 5: Train Model
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 64)
print("[STEP 5/8] Train the Model — Random Forest")
print("─" * 64)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train_scaled, y_train)

print(f"\n  Model: Random Forest Classifier")
print(f"  Trees in the forest: 100")
print(f"  Trained on {X_train.shape[0]} samples with {X_train.shape[1]} features")

# ─────────────────────────────────────────────────────────────
# CELL 6: Evaluate Performance
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 64)
print("[STEP 6/8] Evaluate Performance")
print("─" * 64)

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n  Accuracy: {accuracy:.1%}")
print(f"\n  Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion Matrix chart
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names,
            annot_kws={'size': 16})
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Confusion Matrix — Where Does the Model Get It Wrong?', fontsize=13, fontweight='bold')
plt.tight_layout()
chart2_path = os.path.join(CHART_DIR, 'chart_2_confusion_matrix.png')
plt.savefig(chart2_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Chart saved → chart_2_confusion_matrix.png")

# ─────────────────────────────────────────────────────────────
# CELL 7: Feature Importance
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 64)
print("[STEP 7/8] Feature Importance — Explainability")
print("─" * 64)

importance = model.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': iris.feature_names,
    'Importance': importance
}).sort_values('Importance', ascending=True)

print(f"\n  Feature Rankings:")
for _, row in feature_importance.iterrows():
    bar = "█" * int(row['Importance'] * 40)
    print(f"    {row['Feature']:25s} {row['Importance']:.3f}  {bar}")

plt.figure(figsize=(8, 5))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='steelblue')
plt.xlabel('Importance Score', fontsize=12)
plt.title('Feature Importance — What Drives the Prediction?', fontsize=13, fontweight='bold')
plt.tight_layout()
chart3_path = os.path.join(CHART_DIR, 'chart_3_feature_importance.png')
plt.savefig(chart3_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Chart saved → chart_3_feature_importance.png")

# ─────────────────────────────────────────────────────────────
# CELL 8: Pipeline Architecture Diagram
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 64)
print("[STEP 8/8] Pipeline Architecture")
print("─" * 64)

print("""
  ┌────────────┐   ┌────────────┐   ┌────────────┐
  │ 1. LOAD    │──→│ 2. EXPLORE │──→│ 3. SPLIT   │
  │ 150 rows   │   │ Visualize  │   │ 80/20      │
  │ 4 features │   │ Understand │   │ Train/Test │
  └────────────┘   └────────────┘   └─────┬──────┘
                                          │
                              ┌───────────┴───────────┐
                              ▼                       ▼
                    ┌────────────┐           ┌────────────┐
                    │ 4. SCALE   │           │  (Held out │
                    │ Normalize  │           │   for test)│
                    └─────┬──────┘           └─────┬──────┘
                          ▼                        │
                    ┌────────────┐                 │
                    │ 5. TRAIN   │                 │
                    │ 100 trees  │                 │
                    └─────┬──────┘                 │
                          ▼                        ▼
                    ┌────────────┐           ┌────────────┐
                    │ 6. PREDICT │──────────→│ 7. EVALUATE│
                    └────────────┘           └─────┬──────┘
                                                   ▼
                                            ┌────────────┐
                                            │ 8. EXPLAIN │
                                            └────────────┘
""")

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("=" * 64)
print(f"  PIPELINE COMPLETE")
print(f"  Accuracy: {accuracy:.1%}")
print(f"  Top feature: {feature_importance.iloc[-1]['Feature']}")
print(f"  Charts saved: 3 PNG files")
print("=" * 64)
