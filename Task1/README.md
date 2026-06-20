# Credit Scoring Model

Predicting an individual's creditworthiness (Good/Bad credit risk) using the German Credit dataset, built with Logistic Regression and Random Forest.

> Built as part of the **CodeAlpha Data Science Internship — Task 1**

---

## 📌 Objective

Banks and lenders need a reliable way to assess whether a loan applicant is likely to **repay** or **default**. This project builds a classification model that predicts credit risk based on an applicant's financial and personal history, helping automate and standardize lending decisions.

---

## 📊 Dataset

- **Source:** [Statlog (German Credit Data) — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
- **File used:** `german.data`
- **Size:** 1000 rows × 21 columns
- **Target variable:** Credit risk — originally `1` (Good) / `2` (Bad), remapped to `0` (Good) / `1` (Bad)
- **Class distribution:** 700 Good / 300 Bad (imbalanced)

**Key features include:**
- `duration`, `credit_amount`, `installment_rate`, `age`, `residence_duration`
- `credit_history`, `purpose`, `savings`, `employment`, `housing`, `job`
- `personal_status_sex`, `guarantors`, `property`, `other_installment_plans`
- `telephone`, `foreign_worker`, `number_credits`, `people_liable`

---

## ⚙️ Approach / Workflow

1. **Load & inspect data** — read `german.data`, check shape and target distribution
2. **Data cleaning** — verified no missing values
3. **Feature engineering** — created `credit_per_month` (`credit_amount / duration`) as a derived affordability feature
4. **Encoding** — one-hot encoded all categorical columns (`pd.get_dummies`)
5. **Train/test split** — 80/20 stratified split to preserve class ratio
6. **Scaling** — `StandardScaler` applied for the Logistic Regression model
7. **Model building** — trained two classifiers:
   - Logistic Regression (`liblinear` solver)
   - Random Forest (`class_weight='balanced'` to handle class imbalance)
8. **Evaluation** — Precision, Recall, F1-score, ROC-AUC, Confusion Matrix
9. **Feature importance** — compared LR coefficients vs. RF importances, visualized top 10 RF features
10. **Model persistence** — saved trained model, scaler, and feature list with `joblib`
11. **Inference function** — `score_applicant()` to score a new applicant dict end-to-end

---

## 📈 Results

| Model | Accuracy | ROC-AUC | Precision (Bad) | Recall (Bad) | F1 (Bad) |
|---|---|---|---|---|---|
| Logistic Regression | 0.78 | **0.801** | 0.64 | 0.58 | 0.61 |
| Random Forest | 0.79 | 0.776 | 0.79 | 0.38 | 0.52 |

- **Logistic Regression** generalizes better on ROC-AUC and catches more actual bad-credit cases (higher recall).
- **Random Forest** is more conservative — fewer false positives on bad credit, but misses more actual defaulters.
- Class imbalance (700 Good vs 300 Bad) is the main driver of the recall gap — a good area for further improvement (see below).

**Top predictive features (Random Forest):**
1. `credit_amount`
2. `credit_per_month`
3. `age`
4. `duration`
5. `status_A14` (checking account status)

---

## 🛠️ Tech Stack

- Python, Pandas, NumPy
- scikit-learn (`LogisticRegression`, `RandomForestClassifier`, `StandardScaler`, metrics)
- Matplotlib, Seaborn (feature importance visualization)
- Joblib (model serialization)

---

## 📁 Project Structure

```
credit-scoring-model/
├── CreditScoringModel.ipynb        # Main notebook: EDA → modeling → evaluation
├── german.data                     # Dataset (UCI German Credit Data)
├── credit_scoring_rf_model.pkl     # Saved Random Forest model
├── credit_scoring_scaler.pkl       # Saved StandardScaler
├── credit_scoring_features.pkl     # Saved feature column order
└── README.md
```

---

## 🚀 How to Run

```bash
git clone https://github.com/RaviNamdeoo/credit-scoring-model.git
cd credit-scoring-model
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

1. Download `german.data` from the [UCI repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) and place it in the project root.
2. Open and run `CreditScoringModel.ipynb` cell by cell in Jupyter Notebook / JupyterLab.

---

## 🔍 Sample Prediction

```python
new_applicant = {
    "status": "A11", "duration": 12, "credit_history": "A34", "purpose": "A43",
    "credit_amount": 2500, "savings": "A61", "employment": "A75", "installment_rate": 2,
    "personal_status_sex": "A93", "guarantors": "A101", "residence_duration": 3,
    "property": "A121", "age": 35, "other_installment_plans": "A143", "housing": "A152",
    "number_credits": 1, "job": "A173", "people_liable": 1, "telephone": "A192",
    "foreign_worker": "A201", "credit_per_month": 208.33
}

score_applicant(new_applicant)
# {'prediction': 'Good', 'probability': 0.2}
```

---

## 🔮 Future Improvements

- Handle class imbalance with SMOTE or class-weighted thresholds to improve recall on defaulters
- Try XGBoost / LightGBM and compare against LR and RF
- Add cross-validation instead of a single train/test split
- Build a Streamlit/Flask app for interactive credit scoring
- Hyperparameter tuning (GridSearchCV / Optuna)

---

## 👤 Author

**Ravi Namdeo**
- GitHub: [@RaviNamdeoo](https://github.com/RaviNamdeoo)
- LinkedIn: [linkedin.com/in/ravinamdeo](https://linkedin.com/in/ravinamdeo)
