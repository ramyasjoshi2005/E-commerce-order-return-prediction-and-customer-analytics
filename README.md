# E-Commerce Return Prediction & Customer Analytics

Predict whether a newly placed order will be returned using **only information available at order placement**, then explain associated factors without claiming causation.

This is a focused Data Science project (analysis → statistics → leakage-safe features → model comparison → evaluation → SHAP → simple Flask app). It is aligned with an Amazon Data Scientist-1 Intern skill set: problem framing, data integrity, statistical testing, predictive modeling, validation, and business interpretation.

---

## 1. Project overview

**Question:** Can we predict whether a newly placed e-commerce order will be returned using only information known when the order is placed, and which factors are associated with returns?

**Prediction target:** `returned = 1` (return), `returned = 0` (no return).

**What this repo does:** cleans a labeled order table, engineers point-in-time customer history, runs **one** two-proportion z-test, trains four models on a **temporal split**, locks a threshold on validation, evaluates 2021 once, and serves a Flask Prediction Studio over **real customers**.

---

## 2. Business problem

Returns consume logistics and reverse-supply cost after checkout. A score at **placement** is early enough to route some orders to review or customer-experience follow-up. This project does **not** claim that deploying the model reduces returns by a given percentage. That would require a controlled experiment.

---

## 3. Dataset

### What is used for modeling

| Item | Value |
|---|---|
| Source | Existing labeled Superstore **order** table in this repo (`data/processed/features.csv` → cleaned to `orders_clean.csv`) |
| Grain | One row per `order_id` |
| Orders | 5,009 |
| Customers | 793 |
| Market / country | US / United States only |
| Date range | 2018-01-03 to 2021-12-30 |
| Returns | 296 (5.91% return rate) |

Product **category / sub-category** are not in this labeled order table, so they are not used as model features and are not invented.

`order_priority` is constant (`Medium`) in this table, so it was dropped from the model.

### Why this is not Global Superstore

`data/raw/Global_Superstore2.csv` (2011–2014, 25,035 orders, 147 countries **including India**) and `data/raw/returns.csv` (1,079 rows, IDs through 2015) **do not join**: **0 overlapping Order IDs**, different ID schemes. Using Global Superstore as the modeling dataset would require fabricating return labels. That was not done.

**India** is therefore **not** shown in the app. It exists in the unused Global Superstore file, not in the labeled modeling table.

### Data-integrity fix

The original labeled file had **504 duplicate `order_id` rows**. Those duplicates treated the same order as its own prior history and inflated the apparent return rate (~14.5% → **5.91%** after de-duplication). Duplicates were dropped (`keep='first'` after sorting by date and `order_id`).

---

## 4. Prediction point

The model scores an order **when the customer places it**.

**Not used:** `profit`, `shipping_cost`, `ship_date`, return date, return reason, future orders, or any feature derived from the current order’s return label.

---

## 5. Data leakage prevention

- History for order *t* uses only earlier orders: `(order_date < t)` or `(same date AND order_id < t.order_id)`.
- The current order is **never** included in `previous_order_count`, `previous_return_count`, `historical_return_rate`, or `historical_average_order_value`.
- Train / validation / test are chronological. 2021 is not used for model selection or threshold tuning.
- Training and the Flask app share `src/feature_engineering.py` and `src/predict.py`.

---

## 6. EDA (questions, not chart volume)

Observed return rates (associations only):

- **Overall:** 5.91% (296 / 5,009).
- **US region:** West 11.7% (n=1,611) vs South 2.9% (n=822), East 3.1%, Central 3.3%.
- **Discount:** 0–20% bin 7.5% (n=1,963); 0% bin 4.9%; >40% bin 4.2%.
- **Order value:** >$400 7.8%; ≤$50 3.5%.
- **Prior returns (point-in-time):** 6.3% vs 5.8% (see statistics).

---

## 7. Statistical analysis

**Research question:** Do customers with previous returns have a different subsequent return rate than customers with no previous returns?

| Item | Result |
|---|---|
| Test | Two-sided two-proportion z-test |
| Group A | ≥1 previous return before this order: 49/774 = **6.33%** |
| Group B | 0 previous returns (includes first orders): 247/4,235 = **5.83%** |
| z | 0.541 |
| p-value | 0.589 |
| Difference | +0.50 percentage points |
| 95% CI | −1.36 to +2.35 percentage points |
| Risk ratio | 1.09 |

**Interpretation:** At α = 0.05 we **do not reject** H0. The observed gap is compatible with chance. This is **not** proof that the rates are equal, and it is **not** evidence of causation.

---

## 8. Feature engineering

**Current order (placement-time):** `order_value`, `quantity`, `discount`, `max_discount`, `number_of_products`, `number_of_categories`, `region`, `segment`, `order_month`, `order_day_of_week`, `is_weekend`.

**Point-in-time history:** `previous_order_count`, `previous_return_count`, `historical_return_rate`, `historical_average_order_value`, `days_since_previous_order`, `is_first_order` (first orders use `days_since_previous_order = 9999`).

---

## 9. Model development

**Split (actual years in this table):**

| Split | Period | Orders | Return rate |
|---|---|---:|---:|
| Train | 2018–2019 | 2,007 | 5.68% |
| Validation | 2020 | 1,315 | 5.86% |
| Test | 2021 | 1,687 | 6.22% |

**Validation:** expanding-window `TimeSeriesSplit` (3 folds) **inside training** (mean PR-AUC for the selected model: 0.101). Model **selection** uses the 2020 validation set. Random k-fold was not used as the primary strategy because it would mix later orders into earlier folds.

**Imbalance:** ~6% positives. `class_weight='balanced'` / `scale_pos_weight`. **SMOTE was not used.**

**Models:** majority-class baseline, logistic regression, random forest, XGBoost.

---

## 10. Model evaluation

### Validation comparison (2020), 0.5 cutoff for F1/precision/recall only

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Majority-class baseline | 0.500 | 0.059 | 0.000 | 0.000 | 0.000 |
| **Logistic Regression** | 0.630 | **0.148** | 0.106 | 0.481 | 0.174 |
| Random Forest | 0.674 | 0.118 | 0.167 | 0.026 | 0.045 |
| XGBoost | 0.631 | 0.090 | 0.097 | 0.078 | 0.086 |

**Selected model: logistic regression** — highest **validation PR-AUC**. XGBoost was not assumed to win.

PR-AUC matters because a constant “no return” policy already has ~94% accuracy.

### Threshold

F1 maximized on **validation** over 0.05–0.95: **0.65**, then locked.

### Test (2021, once)

| Metric | Value |
|---|---:|
| ROC-AUC | 0.693 |
| PR-AUC | 0.154 |
| Precision | 0.175 |
| Recall | 0.343 |
| F1 | 0.232 |
| Accuracy (secondary) | 0.858 |
| Brier | 0.170 |
| Confusion | TN 1412, FP 170, FN 69, TP 36 |

**Calibration:** class-weighted logistic regression **overstates** probabilities (Brier 0.17 vs ~0.055 for a constant base-rate predictor). Treat scores as **risk ranks**, not well-calibrated frequencies. The reliability curve is in the app.

After removing label leakage, **predictive lift is real but modest** (test PR-AUC 0.154 vs ~0.062 prevalence). That is the honest result.

---

## 11. Explainability

SHAP / linear attributions on a **validation** sample (not test).

For logistic regression, local bars are **coefficient × transformed feature** (contribution to log-odds). That is the linear-model form of “how this feature moved the score.” **Not causal.**

Largest mean |SHAP| on the validation sample included `number_of_products`, `previous_return_count`, and `region_West`. Multivariate model attribution can still use a feature whose **univariate** rate gap is not statistically significant.

---

## 12. Business insights (observational)

1. **West vs other US regions:** much higher observed return rate in West (11.7%) than South/East/Central (~3%). Association, not a proven cause.
2. **Prior returns:** 6.3% vs 5.8% subsequent return rate; **not statistically significant**.
3. **Discount / value:** mid-low discount bin and higher order-value bins have higher observed return rates. Not evidence that cutting discounts or prices would change returns.

**Recommendation:** use the score to **prioritize review**, not to auto-block customers. Measure any intervention with an A/B test. Do not claim “this model reduces returns by X%.”

---

## 13. Application

Flask + HTML/CSS/JS (no Streamlit):

1. Overview  
2. Data & EDA  
3. Statistics  
4. Model performance  
5. **Prediction Studio** (real customers, history strictly before the chosen date, live model + feature contributions)  
6. Explainability  
7. Interview guide  

---

## 14. Project architecture

```
data/raw/                 # inspected Global Superstore + returns (not joined)
data/processed/           # orders_clean.csv, model_features.csv, returns.db
src/                      # cleaning, features, stats, train, evaluate, SHAP, predict
models/                   # best_model.joblib
artifacts/                # metrics, plots, SQL results
sql/analysis.sql
frontend/                 # index.html, styles.css, app.js
backend/app.py
notebooks/
README.md
requirements.txt
```

---

## 15. How to run

```bash
python -m pip install -r requirements.txt
python -m src.run_pipeline
python backend/app.py
```

Open http://127.0.0.1:5000

Pipeline steps (also runnable individually): `data_cleaning` → `feature_engineering` → `eda` → `statistics` → `sql_analysis` → `train` → `evaluate` → `explainability` → `insights`.

---

## 16. Limitations

- US Superstore only; no India in the labeled table.
- No product category on the labeled grain.
- Weak positive class (~6%) and modest PR-AUC.
- Probabilities are not well calibrated under class weighting.
- No experiment on business impact.
- Global Superstore returns file is unusable without a valid join.

---

## 17. Future improvements

- A matching returns file (or line-item categories) without inventing labels.
- Probability calibration fitted only on training/validation.
- A randomized test of any review / outreach policy.

---

## How I would explain this project in an interview

“I built an order-placement return model on a labeled US Superstore table. I first checked data integrity: Global Superstore files in the repo don’t join to the returns file, so I did not swap datasets or invent labels. I also dropped duplicate order IDs that had been leaking the current return into customer history and inflating the return rate.

I engineered placement-time order features and point-in-time history. I tested whether prior returns predicted later returns with a two-proportion z-test; the difference was small and not significant.

I trained a majority baseline, logistic regression, random forest, and XGBoost on 2018–2019, selected on 2020 PR-AUC, locked a 0.65 F1 threshold on validation, and evaluated 2021 once. Logistic regression won; I did not assume XGBoost would. Lift is modest, and class-weighted probabilities are not well calibrated, so I treat the output as a ranking score for review, not a causal lever and not a guaranteed return reduction.”

---

## How I would explain this in 2–3 minutes

Use the paragraph above. The implementation matches that story: leakage fix, non-significant z-test, LR selected on validation PR-AUC, locked threshold, SHAP/linear attributions, Flask studio on real customers.
