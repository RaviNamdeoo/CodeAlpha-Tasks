
# Disease Prediction from Medical Data

A binary classification project that predicts the likelihood of diabetes in patients using clinical/diagnostic measurements, built with Logistic Regression and Random Forest.

## Overview

This project uses the **Pima Indians Diabetes Dataset** to predict whether a patient has diabetes based on diagnostic measurements such as glucose level, BMI, blood pressure, and age. Two classification models are trained and compared, with performance evaluated using accuracy, classification report, confusion matrix, and ROC-AUC.

## Dataset

- **Source:** [Pima Indians Diabetes Dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)
- **Size:** 768 patient records, 8 features + 1 target
- **Target:** `Outcome` (0 = No Diabetes, 1 = Diabetes)
- **Features:**
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

## Approach

1. Loaded and explored the dataset (shape, datatypes, class balance)
2. Split data into train/test sets (80/20, stratified on target)
3. Standardized features using `StandardScaler`
4. Trained two models:
   - Logistic Regression
   - Random Forest Classifier (100 estimators)
5. Evaluated both models using accuracy, classification report, confusion matrix, and ROC-AUC

## Results

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | 71.4%    |
| Random Forest        | 76.0%    |

**Random Forest — Classification Report**

| Class        | Precision | Recall | F1-score |
|--------------|-----------|--------|----------|
| No Diabetes  | 0.79      | 0.85   | 0.82     |
| Diabetes     | 0.68      | 0.59   | 0.63     |

- **ROC-AUC Score (Random Forest):** 0.815

## Key Insight

Random Forest outperformed Logistic Regression on raw accuracy, but recall on the "Diabetes" class (0.59) is noticeably lower than recall on "No Diabetes" (0.85) — meaning the model misses a meaningful share of actual diabetes cases. In a medical screening context, this is the metric worth improving most, since a missed positive case carries more real-world cost than a false alarm.

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook Disease_Prediction_from_MedicalData.ipynb
```

## Future Improvements

- Address class imbalance (500 vs. 268 samples) with techniques like SMOTE or class weighting, to improve recall on the diabetes class
- Try additional models (XGBoost, LightGBM) and compare
- Hyperparameter tuning via GridSearchCV/Optuna
- Feature importance analysis to identify which diagnostic measures matter most

## Author

**Ravi Namdeo**
- GitHub: [RaviNamdeoo](https://github.com/RaviNamdeoo)
- LinkedIn: [RaviNamdeo](https://linkedin.com/in/ravinamdeo)
