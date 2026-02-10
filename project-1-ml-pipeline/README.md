# Project 1: ML Pipeline — Iris Classifier

A complete 8-step ML pipeline demonstrating the full arc from raw data to explainable predictions. Built from the Solutions Architect perspective — focused on pipeline architecture, data quality, and explainability rather than model tuning.

## Architecture

```
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
```

## Results

- **Model:** Random Forest (100 trees)
- **Accuracy:** 100% on test set (30 samples)
- **Top Feature:** Petal length (44% importance)
- **Key Insight:** Petal measurements drive 86% of predictions; sepal width contributes only 3%

## Key SA Concepts Demonstrated

- **Data leakage prevention** — fit_transform on train only, transform on test
- **Train/test split** — model never sees evaluation data during training
- **Feature importance** — explainability layer for stakeholder communication
- **Confusion matrix** — understanding *where* the model fails, not just *how often*
- **Pipeline architecture** — the model is one step; most value is in everything around it

## Run It

```bash
pip install -r requirements.txt
python run_pipeline.py
```

Charts are saved as PNG files in this directory.

## Files

| File | Purpose |
|------|---------|
| `run_pipeline.py` | Complete pipeline script (terminal-friendly) |
| `project-1-ml-pipeline.ipynb` | Jupyter notebook version |
| `requirements.txt` | Python dependencies |
| `chart_1_feature_distributions.png` | Feature exploration visualization |
| `chart_2_confusion_matrix.png` | Model evaluation heatmap |
| `chart_3_feature_importance.png` | Explainability chart |

## Status

✅ Complete — Day 1, Session 2
