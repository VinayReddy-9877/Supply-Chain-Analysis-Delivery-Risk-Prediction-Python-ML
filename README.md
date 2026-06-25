# Supply Chain Analysis & Delivery Risk Prediction

> **End-to-end data analytics and machine learning project** that diagnoses *why* 54.7% of orders across a global e-commerce supply chain arrive late — and predicts *which* future orders are at risk before they ship.

---

## Business Problem

DataCo's supply chain spans five global markets and four shipping modes. Despite $35.2M in total revenue, more than half of all orders are delivered late — driving re-shipment costs, customer churn, and eroded profit margins. Leadership needed three questions answered with data:

1. **Where** are delays concentrated — which regions, shipping modes, departments?
2. **Why** are they happening — which factors are statistically proven drivers vs visual noise?
3. **Can delays be predicted** at order placement time, before any shipment occurs?

---

## Project Highlights

| Metric | Value |
|--------|-------|
| Orders Analysed | 172,765 |
| Current Delay Rate | **54.7%** (94,523 orders) |
| Revenue at Risk | **$19.25M** |
| Revenue Protected (10% delay reduction) | **$1.93M** |
| Best Model | Random Forest — **F1 = 0.758 \| AUC = 0.834** |
| #1 Validated Driver | Shipping Mode **(chi² = 51,076, p < 0.001)** |

---

## Repository Structure

```
├── Supply_Chain_Analysis___Delivery_Risk_Prediction_Project.ipynb   # Full notebook
├── DataCoSupplyChainDataset.csv                                     # Source dataset
├── Supply_Chain_Analysis_Report.docx                                # Full project report
└── README.md
```

---

## Dataset

- **DataCo Global Supply Chain Dataset** — 180,519 raw orders × 53 columns
- Covers orders, customers, products, shipping, and financials across 5 global markets
- After removing cancelled orders: **172,765 orders** used for analysis

---

## Tools & Libraries

| Category | Library |
|----------|---------|
| Language | Python 3 (Jupyter Notebook) |
| Data Wrangling | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Statistics | `scipy` — chi-square tests |
| Machine Learning | `scikit-learn` — LR, DT, RF, cross-validation, RandomizedSearchCV |
| Gradient Boosting | `xgboost` |
| Class Imbalance | `imbalanced-learn` — SMOTE |
| Explainability | `shap` — TreeExplainer |

---

## Methodology — 8 Stages

```
1. Data Cleaning          → Removed 29 PII/irrelevant columns, 7,754 cancelled orders
2. Feature Engineering    → 7 new features including binary target Is_Delayed (from raw dates)
3. EDA                    → KPI dashboard, profitability, 6-dimension bottleneck analysis
4. Time-Based Analysis    → Monthly, day-of-week, hourly delay patterns (all flat → no seasonality)
5. Statistical Validation → Chi-square independence tests — separates real drivers from noise
6. Root Cause Diagnostic  → Investigated 100%/0% anomaly in First Class / Same Day shipping
7. Predictive Modelling   → 4 classifiers + SMOTE + 5-fold CV + hyperparameter tuning + SHAP
8. Business Impact        → Findings translated to $19.25M exposure and $1.93M recovery scenario
```

---

## Key Findings

**1 — Shipping Mode is the dominant driver (chi² = 51,076, p < 0.001)**
Standard Class carries a 39.8% delay rate — the highest among modes with genuine variability. First Class (100%) and Same Day (0%) are fixed scheduling artifacts, not real performance signals, confirmed after ruling out five alternative hypotheses.

**2 — No seasonal effect exists**
Quarterly delay rates are flat (54.5%–54.8%). Recommendations targeting Q4 surge, weekend staffing, or time-of-day order management are not supported by this data.

**3 — Department Name is NOT a significant driver (p = 0.51)**
Despite appearing in EDA bar charts, department-level delay differences are statistically indistinguishable from random variation. No department-level inventory action is warranted.

**4 — Geography matters but is secondary**
Regional delays are significant (chi² = 86.1, p < 0.001) but concentrated — Central Africa (58.7%) and Southeast Asia (56.0%) drive most of the regional variance. Within-region analysis confirms shipping mode explains regional differences, not inherent geographic factors.

**5 — Feature importance needs independent validation**
`order_hour` ranked #1 by raw Random Forest importance (0.226) despite a 5.2-point delay-rate spread. It is a tree-model artifact caused by high cardinality. SHAP and chi-square both confirm it has negligible real-world impact. `Days for shipment (scheduled)` is the true top predictor, confirmed by three independent methods.

---

## Machine Learning Results

| Model | Accuracy | Precision | Recall | F1 | CV F1 (5-fold) |
|-------|----------|-----------|--------|----|----------------|
| Logistic Regression | 0.673 | 0.771 | 0.573 | 0.657 | 0.637 ± 0.005 |
| Decision Tree | 0.726 | 0.882 | 0.576 | 0.697 | 0.687 ± 0.005 |
| **Random Forest ★** | **0.744** | 0.784 | **0.734** | **0.758** | **0.748 ± 0.003** |
| XGBoost | 0.729 | 0.864 | 0.598 | 0.707 | 0.703 ± 0.006 |
| RF (Tuned) | 0.744 | 0.812 | 0.691 | 0.747 | — |
| XGBoost (Tuned) | 0.745 | 0.843 | 0.656 | 0.738 | — |

**★ Best Model: Random Forest Baseline — F1 = 0.758, AUC = 0.834**

Tuning improved XGBoost by +0.031 F1 but did not improve Random Forest, indicating its default configuration was already near-optimal for this feature set.

---

## Business Impact

| Metric | Value |
|--------|-------|
| Current Delay Rate | 54.7% |
| Total Delayed Orders | 94,523 |
| Revenue Tied to Delayed Orders | $19.25M |
| Profit Tied to Delayed Orders | $2.06M |
| Revenue Protected — 10% Delay Reduction | **$1.93M** |

---

## Recommendations

| Priority | Root Cause | Action | Target KPI |
|----------|-----------|--------|------------|
| **P1 — Critical** | Standard Class: 39.8% delay rate (chi² = 51,076) | Renegotiate carrier SLAs; optimise warehouse dispatch for Standard Class; auto-upgrade at-risk orders at pick time | Reduce Standard Class delay rate by 10 pts within 2 quarters |
| **P2 — High** | High-delay regions (chi² = 86.1) | Partner with 3PL providers in Central Africa, Southeast Asia, Eastern Europe | All regions below 20% delay rate within 4 quarters |
| **P3 — Medium** | Customer Segment (chi² = 6.74, p = 0.034 — small effect) | A/B test segment-specific SLAs before committing resources | Validate effect size before setting a KPI |

**Removed from scope (not supported by data):** Q4 seasonal surge actions, large-order routing, department-level inventory changes.

---

## Notebook Structure

```
Section 1  — Imports & Configuration
Section 2  — Data Loading & Initial Exploration
Section 3  — Data Cleaning (29 columns removed, cancellations filtered)
Section 4  — Feature Engineering (7 new features + Is_Delayed target)
Section 5  — Visualisation: Profitability Distribution
Section 6  — KPI Dashboard
Section 7  — Delay Distribution & Profit vs Delay Analysis
Section 8  — Six-Dimension Bottleneck Detection
Section 9  — Shipping Mode / Region / Department Deep-Dive
Section 10 — Seasonal, Quarter, Weekend, Order-Size Analysis
Section 11 — Root Cause Diagnostic (First Class / Same Day anomaly)
Section 12 — Time-Based Analysis (Month / Day / Hour)
Section 13 — Statistical Validation (Chi-Square Tests)
Section 14 — Target Leakage Validation
Section 15 — Encoding & Train/Test Split
Section 16 — SMOTE Class Balancing
Section 17 — 5-Fold Cross-Validation
Section 18 — Model Training: Logistic Regression
Section 19 — Model Training: Decision Tree
Section 20 — Model Training: Random Forest
Section 21 — Model Training: XGBoost
Section 22 — Model Comparison Chart
Section 23 — Confusion Matrices & ROC Curves
Section 24 — Hyperparameter Tuning (RF + XGBoost, RandomizedSearchCV)
Section 25 — Baseline vs Tuned Comparison
Section 26 — SHAP Explainability (XGBoost Tuned)
Section 27 — Feature Importance Analysis (Random Forest)
Section 28 — Business Impact Assessment
Section 29 — Key Insights Summary
Section 30 — Business Recommendations & Action Plan
```

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/supply-chain-analysis.git
cd supply-chain-analysis

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost imbalanced-learn shap jupyter

# 3. Place the dataset in the expected path
#    The notebook reads from: /content/DataCoSupplyChainDataset.csv
#    Update the path in Cell 6 if running locally

# 4. Launch Jupyter and run all cells
jupyter notebook Supply_Chain_Analysis___Delivery_Risk_Prediction_Project.ipynb
```

** K Vinay Reddy**
Email: k.vinayreddy9877@gmail.com
[LinkedIn] (www.linkedin.com/in/k-vinay-reddy-059634285)
