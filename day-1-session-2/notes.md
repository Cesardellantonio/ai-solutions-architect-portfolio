# Day 1, Session 2 — The Architect's Role + Your First ML Pipeline

> **Duration:** 1 hour
> **Style:** Guided learning (20 min) + Hands-on build (40 min)
> **Goal:** Understand the ML Solutions Architect's responsibilities, then build a real ML pipeline
> **Environment:** Google Colab (Python)

---

## Part 1: The ML Solutions Architect's Role
*Read time: ~20 minutes*

### Where You Fit

In Session 1, we said: *"The data scientist builds the model. The Solutions Architect builds the system."*

Now let's get specific about what "the system" means. An ML project has five phases. Here's who owns what:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ML PROJECT LIFECYCLE                         │
│                                                                 │
│  1. PROBLEM       2. DATA          3. MODEL        4. DEPLOY   │
│  FRAMING          ENGINEERING      DEVELOPMENT     & SERVE     │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│  │ Business │    │ Pipelines│    │ Training │    │ Infra    │ │
│  │ Problem  │    │ Storage  │    │ Tuning   │    │ APIs     │ │
│  │ Success  │    │ Quality  │    │ Evaluate │    │ Scale    │ │
│  │ Metrics  │    │ Features │    │ Select   │    │ Monitor  │ │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│                                                                 │
│  ◄── SA LEADS ──► ◄── SA + DS ──► ◄── DS LEADS ──► ◄ SA LEADS │
│                                                                 │
│                    5. MONITOR & MAINTAIN                        │
│                    ┌──────────────────┐                         │
│                    │ Drift detection  │                         │
│                    │ Retraining       │                         │
│                    │ Cost tracking    │                         │
│                    │ Business impact  │                         │
│                    └──────────────────┘                         │
│                    ◄──── SA LEADS ────►                         │
└─────────────────────────────────────────────────────────────────┘
```

Notice the pattern: **the Solutions Architect leads 3 out of 5 phases.** The data scientist leads model development. Everything else — framing the problem, engineering the data, deploying the solution, monitoring it in production — that's you.

### The Six Responsibilities

**1. Translate Business Problems into ML Problems**

The VP says: *"We're losing customers."* That's a business problem.

You translate it to: *"Given a customer's last 90 days of behavior data, predict the probability they will cancel within 30 days."* That's an ML problem.

This translation is the hardest part. Get it wrong, and the data scientist builds a perfect model for the wrong question.

| Business Problem | Bad ML Translation | Good ML Translation |
|---|---|---|
| "We're losing customers" | "Predict churn" (too vague) | "Predict 30-day churn probability using 90 days of usage data" |
| "Sales is slow" | "Predict revenue" (too broad) | "Score pipeline deals by close probability + expected value, updated weekly" |
| "Support is overwhelmed" | "Automate support" (too ambitious) | "Classify incoming tickets into 5 categories to auto-route to correct team" |

**2. Design the Data Architecture**

The model is only as good as the data it eats. You design:

- **Where data lives** — Data lake? Data warehouse? Feature store?
- **How data flows** — Batch ETL? Real-time streaming? Event-driven?
- **Data quality** — Validation rules, missing value handling, schema enforcement
- **Feature engineering** — Which raw data transforms into useful model inputs?

This is usually 60-80% of the project timeline. Not glamorous. Absolutely critical.

**3. Choose the Infrastructure**

| Decision | Options | You Choose Based On |
|---|---|---|
| Training compute | Local GPU, Cloud GPU, Managed service | Budget, data size, team skill |
| Model serving | REST API, batch job, edge deployment | Latency requirement, scale |
| Storage | S3, BigQuery, Snowflake, feature store | Query patterns, cost, existing stack |
| Orchestration | Airflow, Step Functions, Kubeflow | Complexity, team familiarity |
| Monitoring | MLflow, Weights & Biases, custom | What metrics matter |

**4. Define Success Metrics (Before Building)**

This is where most ML projects fail. The data scientist optimizes for model accuracy. But does 95% accuracy matter if the business impact is zero?

```
MODEL METRICS (Data Scientist cares)     BUSINESS METRICS (You care)
├── Accuracy: 94%                        ├── Churn reduced by: 12%
├── Precision: 0.91                      ├── Revenue saved: $2.3M/year
├── Recall: 0.87                         ├── Support tickets auto-routed: 73%
├── F1 Score: 0.89                       ├── Sales rep time saved: 4hrs/week
└── AUC-ROC: 0.96                        └── Customer satisfaction: +8 NPS
```

You define both. You connect them: *"If recall drops below 0.80, we miss too many at-risk customers, and revenue savings drop below $1M — not worth the infrastructure cost."*

**5. Manage the ML Lifecycle**

Models aren't code you deploy once. They degrade.

- **Data drift** — Customer behavior changes. The patterns the model learned become stale.
- **Model drift** — Predictions get worse over time even if the model doesn't change.
- **Concept drift** — The relationship between inputs and outputs fundamentally changes (e.g., a pandemic shifts buying behavior overnight).

You build the system that detects drift, triggers retraining, and validates new models before they replace old ones.

**6. Communicate Across Teams**

You're the translator. You speak:
- **Business** to executives: ROI, timelines, risk
- **Technical** to data scientists: architecture, constraints, data access
- **Operational** to engineering: deployment, monitoring, SLAs
- **Product** to designers: what the model can and can't do, confidence levels, edge cases

No one else on the team spans all four languages. That's your superpower.

### The Architect's Decision Framework

For every ML project, run through this checklist:

```
□ PROBLEM FRAMING
  - Is this actually an ML problem? (Or do rules work?)
  - What's the prediction target?
  - What does "good enough" accuracy look like?
  - What's the cost of a wrong prediction?

□ DATA READINESS
  - Does the data exist?
  - Is it accessible? (Legal, technical, political barriers)
  - Is there enough of it? (Hundreds? Thousands? Millions?)
  - Is it clean? (Or will 80% of time go to cleaning?)

□ INFRASTRUCTURE
  - What's the latency requirement?
  - What's the scale? (Predictions per second)
  - What's the budget?
  - What existing systems must this integrate with?

□ DEPLOYMENT
  - How will the model be served?
  - What happens when it's wrong?
  - Who monitors it?
  - When does it retrain?

□ SUCCESS
  - Business metric tied to model metric
  - Timeline to measure impact
  - Kill criteria (when to shut it down)
```

---

## Part 2: Build Your First ML Pipeline (Hands-On)
*Build time: ~40 minutes*

### What We're Building

A complete ML pipeline that:
1. Loads a real dataset (Iris — classic, clean, perfect for learning)
2. Explores and visualizes the data
3. Prepares features for the model
4. Trains a classifier
5. Evaluates performance
6. Visualizes results

This is Project 1 in your portfolio: **Basic ML Pipeline**.

### Setup

1. Go to [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. Name it: `project-1-ml-pipeline.ipynb`

### The Code (Step by Step)

Copy each cell into Colab. Read the comments. Run each cell. Understand what happens before moving to the next.

**Cell 1: Import Libraries**
```python
# Every ML project starts with these
import pandas as pd                  # Data manipulation (think: spreadsheets in code)
import numpy as np                   # Math operations (think: calculator on steroids)
import matplotlib.pyplot as plt      # Visualization (think: chart maker)
import seaborn as sns                # Pretty visualizations (think: chart maker, but prettier)

from sklearn.datasets import load_iris          # Our dataset
from sklearn.model_selection import train_test_split  # Split data for training/testing
from sklearn.preprocessing import StandardScaler      # Normalize features (remember Session 1?)
from sklearn.ensemble import RandomForestClassifier   # Our model (a collection of decision trees)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Libraries loaded. Ready to build.")
```

**Cell 2: Load and Explore the Data**
```python
# Load the Iris dataset — 150 flowers, 4 measurements each, 3 species
iris = load_iris()

# Convert to a DataFrame (like a spreadsheet)
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target  # Add the label (0, 1, or 2 = three species)
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# What does our data look like?
print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nFeatures (inputs): {iris.feature_names}")
print(f"Labels (outputs): {iris.target_names}")
print(f"\nFirst 5 rows:")
df.head()
```

**Cell 3: Visualize the Data (Architect's Eye)**
```python
# As a Solutions Architect, you need to understand the data BEFORE modeling
# This tells you: are the classes separable? Are features useful?

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Iris Dataset — Feature Distributions by Species', fontsize=14)

for i, feature in enumerate(iris.feature_names):
    ax = axes[i // 2, i % 2]
    for species_id, species_name in enumerate(iris.target_names):
        subset = df[df['species'] == species_id]
        ax.hist(subset[feature], alpha=0.6, label=species_name, bins=15)
    ax.set_xlabel(feature)
    ax.set_ylabel('Count')
    ax.legend()

plt.tight_layout()
plt.show()

# WHAT TO NOTICE: If the colors separate cleanly, that feature is useful.
# If they overlap completely, that feature won't help much.
```

**Cell 4: Prepare the Data (The Pipeline Begins)**
```python
# STEP 1: Separate features (X) from labels (y)
X = df[iris.feature_names]  # The 4 measurements (inputs)
y = df['species']            # The species to predict (output)

# STEP 2: Split into training and testing sets
# 80% for training, 20% for testing — the model never sees the test data during training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # random_state = reproducible results
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples")

# STEP 3: Normalize features (remember the feature store discussion?)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Learn scaling from training data
X_test_scaled = scaler.transform(X_test)          # Apply same scaling to test data

# WHY: fit_transform on training, just transform on testing.
# If you fit on test data too, you're leaking future information into the model.
# This is called "data leakage" — a common mistake the SA should catch.
print("\nFeatures normalized. Pipeline step 1 complete.")
```

**Cell 5: Train the Model**
```python
# Random Forest = a collection of decision trees that vote on the answer
# Think of it as a panel of experts. Each tree sees a random subset of data,
# makes its own decision, and the majority vote wins.

model = RandomForestClassifier(
    n_estimators=100,    # 100 decision trees in our forest
    random_state=42      # Reproducible results
)

# Train the model — this is where it learns patterns
model.fit(X_train_scaled, y_train)

print("Model trained on 120 samples.")
print("Now let's see how it performs on the 30 samples it's never seen...")
```

**Cell 6: Evaluate Performance**
```python
# Make predictions on the test set (data the model has never seen)
y_pred = model.predict(X_test_scaled)

# How accurate?
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.1%}")
print(f"\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion Matrix — shows exactly where the model gets confused
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix — Where Does the Model Get It Wrong?')
plt.show()

# ARCHITECT'S QUESTION: Is this accuracy good enough for the business problem?
# 100% on Iris is easy. Real-world problems are messier.
```

**Cell 7: Feature Importance (The Explainability Layer)**
```python
# Remember from Session 1: you chose Option C — show the user WHY
# Random Forests can tell us which features mattered most

importance = model.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': iris.feature_names,
    'Importance': importance
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(8, 5))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='steelblue')
plt.xlabel('Importance Score')
plt.title('Feature Importance — What Drives the Prediction?')
plt.show()

# ARCHITECT INSIGHT: If one feature dominates, the model is simple.
# If importance is spread evenly, the relationships are complex.
# This shapes your architecture decisions (latency, compute needs).
```

**Cell 8: The Full Pipeline Diagram**
```python
# Let's visualize what we just built — this is the architecture artifact

pipeline_text = """
╔══════════════════════════════════════════════════════════════════╗
║                   ML PIPELINE ARCHITECTURE                      ║
║                   Project 1: Iris Classifier                    ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  ┌────────────┐   ┌────────────┐   ┌────────────┐              ║
║  │ 1. DATA    │──→│ 2. PREPARE │──→│ 3. SPLIT   │              ║
║  │ Load Iris  │   │ Clean &    │   │ 80% Train  │              ║
║  │ 150 rows   │   │ Explore    │   │ 20% Test   │              ║
║  │ 4 features │   │ Visualize  │   │            │              ║
║  └────────────┘   └────────────┘   └─────┬──────┘              ║
║                                          │                      ║
║                              ┌───────────┴───────────┐          ║
║                              ▼                       ▼          ║
║                    ┌────────────┐           ┌────────────┐      ║
║                    │ 4. SCALE   │           │ Hold for   │      ║
║                    │ Normalize  │           │ testing    │      ║
║                    │ features   │           │ (unseen)   │      ║
║                    └─────┬──────┘           └─────┬──────┘      ║
║                          ▼                        │              ║
║                    ┌────────────┐                  │              ║
║                    │ 5. TRAIN   │                  │              ║
║                    │ Random     │                  │              ║
║                    │ Forest     │                  │              ║
║                    │ (100 trees)│                  │              ║
║                    └─────┬──────┘                  │              ║
║                          ▼                        ▼              ║
║                    ┌────────────┐           ┌────────────┐      ║
║                    │ 6. PREDICT │──────────→│ 7. EVALUATE│      ║
║                    │ on test    │           │ Accuracy   │      ║
║                    │ data       │           │ Confusion  │      ║
║                    │            │           │ Matrix     │      ║
║                    └────────────┘           └─────┬──────┘      ║
║                                                   ▼              ║
║                                            ┌────────────┐       ║
║                                            │ 8. EXPLAIN │       ║
║                                            │ Feature    │       ║
║                                            │ Importance │       ║
║                                            └────────────┘       ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
print(pipeline_text)
```

### After Building — Save Your Work

1. In Colab: File → Download → Download .ipynb
2. Save to: `ai-solutions-architect-portfolio/project-1-ml-pipeline/`
3. This is your first portfolio artifact

---

## Part 3: Self-Assessment & Wrap-Up

### Can You Explain These?

- [ ] **The 5 phases of an ML project** — which does the SA lead?
- [ ] **Problem framing** — why is "predict churn" a bad ML problem definition?
- [ ] **Data leakage** — what is it and why should the SA catch it?
- [ ] **Train/test split** — why do we hide data from the model?
- [ ] **Feature importance** — why does the SA care about this?

### What You Built Today

A complete ML pipeline with 8 steps:
1. Load data
2. Explore and visualize
3. Split train/test
4. Normalize features
5. Train a Random Forest classifier
6. Make predictions
7. Evaluate with confusion matrix
8. Explain with feature importance

That's not a toy. That's the skeleton of every production ML system. Real systems add complexity (more data, more features, more models, more monitoring) — but the bones are the same.

### Key Insight

> **The ML pipeline is the architecture. The model is just one step in it. Most of the value (and most of the work) is in everything around the model: data preparation, feature engineering, evaluation, and monitoring. That's the Solutions Architect's domain.**

---

## Resources

- **Book:** *The ML Solutions Architect Handbook* by David Ping — Chapter 1
- **Google Colab:** [colab.research.google.com](https://colab.research.google.com)
- **Iris Dataset:** Included in scikit-learn, no download needed
- **Session 3 Preview:** Refine the pipeline + push to GitHub
