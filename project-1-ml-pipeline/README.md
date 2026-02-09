# Project 1: Basic ML Pipeline

A classification pipeline demonstrating data ingestion, model training, evaluation, and bias detection — applied to a SaaS churn prediction scenario.

## Architecture

```
[Data Source] → [Data Ingestion] → [Preprocessing] → [Model Training] → [Evaluation] → [Bias Detection]
     │                                                       │                              │
  UCI Iris /                                          scikit-learn                    Fairness metrics
  custom dataset                                      classifier                     & ethical checks
```

> Full architecture diagram: `diagrams/pipeline-architecture.png`

## Business Value

**Scenario:** SaaS company predicting customer churn to reduce revenue loss. This pipeline demonstrates how an ML solution architect designs the data flow from raw customer data to actionable predictions, with built-in ethical safeguards.

## Setup

```bash
# Open in Google Colab or run locally
pip install -r requirements.txt
python pipeline.py
```

## Key Concepts

- ML pipeline architecture (ETL → Train → Evaluate)
- Classification algorithms (decision trees, random forests)
- Bias detection and ethical AI checks
- Data flow diagramming for stakeholder communication

## Status

🔲 In Progress
