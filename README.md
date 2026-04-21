# 🕸 Fraud Detection with Graph Convolutional Networks (PyTorch)

**Anomaly Detection in Financial Transaction Networks Using Dask for Scalability**

> Topic 62 (★★★★) · Business Analytics in Python  
> Tagirov Islam Rashidovich · Group 15.27д-би06/25м · REU Plekhanov · 2026

---

## Overview

This project implements a complete fraud detection pipeline using **Graph Convolutional Networks (GCN)** on the Credit Card Fraud Detection dataset (284,807 transactions). Transactions are represented as a graph where each transaction is a node connected to its k-nearest neighbors in feature space, enabling the GCN to capture relational patterns that tabular methods miss.

## Key Results

| Model | Precision | Recall | F1 Score | AUC-ROC |
|-------|-----------|--------|----------|---------|
| Logistic Regression | — | — | — | — |
| XGBoost | — | — | — | — |
| **GCN (3-layer)** | **—** | **—** | **best** | **best** |

*(Values computed during notebook execution)*

## Technology Stack

| Tool | Role |
|------|------|
| **Dask** | Scalable data loading and preprocessing (lazy evaluation, parallel I/O) |
| **PyTorch** | Deep learning framework (dynamic computation graph) |
| **PyTorch Geometric** | GCN implementation (GCNConv, sparse ops, Data object) |
| **NetworkX** | Graph construction and analysis |
| **Scikit-learn** | Baselines (LogReg), preprocessing (StandardScaler), metrics |
| **XGBoost** | Gradient boosting baseline |
| **Matplotlib / Seaborn / Plotly** | Visualization |

## Project Structure

```
├── fraud_gcn_notebook.ipynb   # Full pipeline: EDA → Graph → GCN → Results
├── dashboard.html             # Interactive analytics dashboard
├── README.md                  # This file
└── requirements.txt           # Python dependencies
```

## How to Run

### Option 1: Google Colab (recommended)
1. Open `fraud_gcn_notebook.ipynb` in Google Colab
2. Run all cells — dataset downloads automatically via `kagglehub`
3. Results and visualizations generate inline

### Option 2: Local
```bash
pip install -r requirements.txt
jupyter notebook fraud_gcn_notebook.ipynb
```

## Dashboard

Open `dashboard.html` in any browser — interactive analytics with charts, KPI cards, and model comparison. No server required.

## Dataset

**Credit Card Fraud Detection** (Kaggle)  
- 284,807 transactions · 30 features (V1–V28 + Amount + Time)  
- Binary target: Class (0 = legit, 1 = fraud)  
- Fraud rate: 0.172% (492 fraudulent transactions)  
- Source: [kaggle.com/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Methodology

1. **EDA** — class distribution, amount distributions, temporal patterns, correlation analysis
2. **Dask preprocessing** — lazy CSV loading, parallel computation, filter-first pattern
3. **Graph construction** — k-NN graph (k=5) from standardized features, ~5K nodes, ~50K edges
4. **Baseline models** — Logistic Regression and XGBoost (tabular, no graph structure)
5. **GCN training** — 3-layer GCNConv with weighted cross-entropy, early stopping, Adam optimizer
6. **Evaluation** — F1, Precision, Recall, AUC-ROC, confusion matrices, ROC/PR curves

## References

1. Kipf & Welling (2017). Semi-supervised classification with GCN. ICLR.
2. Hamilton et al. (2017). Inductive representation learning on large graphs. NeurIPS.
3. Lopez-Rojas et al. (2016). PaySim: A financial mobile money simulator. EMSS.
4. Lin et al. (2017). Focal loss for dense object detection. ICCV.

---

*REU Plekhanov · Department of Computer Science · 2026*
