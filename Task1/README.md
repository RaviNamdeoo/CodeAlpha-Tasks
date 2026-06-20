# 💳 CodeAlpha Credit Scoring Model

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/bhanukumardev/CodeAlpha_CreditScoringModel) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/) [![HTML](https://img.shields.io/badge/HTML-E34F26?logo=html5&logoColor=white)](https://html.spec.whatwg.org/) [![Stars](https://img.shields.io/github/stars/bhanukumardev/CodeAlpha_CreditScoringModel?style=social)](https://github.com/bhanukumardev/CodeAlpha_CreditScoringModel/stargazers)

> Credit Scoring Model project for Code Alpha Internship — ML-powered credit risk assessment using German Credit Data.

## 📹 Demo

**[📊 View LinkedIn Post →](https://www.linkedin.com/posts/bhanu-kumar-dev-97b820313_machinelearning-creditscoring-datascience-activity-7335941965395492865-vW-f?utm_source=share&utm_medium=member_desktop&rcm=ACoAAE-td28BKSK7mi1hQgrYDtXPTq_qe8XRr18)**

## 🚀 Overview

A comprehensive credit risk prediction system built using **machine learning** algorithms. This project demonstrates:

- **Predictive Modeling** - Logistic Regression & Random Forest classifiers
- **Feature Analysis** - Identifying key credit risk indicators
- **Model Evaluation** - Precision, Recall, F1-score, ROC-AUC metrics
- **Interpretability** - Feature importance visualization

Built as part of **Code Alpha Data Science Internship**.

## ✨ Features

- 📊 **Multiple ML Models** - Logistic Regression, Random Forest
- 🔍 **Feature Engineering** - Data preprocessing and transformation
- 🎯 **High Accuracy** - Optimized hyperparameters
- 📈 **Comprehensive Evaluation** - ROC curves, confusion matrices
- 📉 **Feature Importance** - Understand what drives predictions
- 💾 **Model Persistence** - Saved model for deployment

## 📊 Dataset

**UCI German Credit Data**

- **Source:** UCI Machine Learning Repository
- **Samples:** 1000 credit applications
- **Features:** 20 attributes (age, job, credit history, savings, etc.)
- **Target:** Binary classification (Good/Bad credit risk)

## 🛠️ Tech Stack

### Machine Learning

- **Python 3.8+**
- **Scikit-learn** - ML algorithms and evaluation
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Visualization

### Tools

- **Jupyter Notebook** - Interactive development
- **Pickle** - Model serialization
- **HTML Slides** - Presentation export

## 📦 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook
- Git

### Clone Repository

```bash
git clone https://github.com/bhanukumardev/CodeAlpha_CreditScoringModel.git
cd CodeAlpha_CreditScoringModel
```

### Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Run Jupyter Notebook

```bash
jupyter notebook CodeAlpha_CreditScoringModel.ipynb
```

## 🎯 Usage Example

### Load Trained Model

```python
import pickle
import pandas as pd

# Load the saved model
with open('credit_scoring_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare new data
new_applicant = pd.DataFrame({
    'age': [35],
    'job': [2],
    'credit_history': [1],
    'savings': [4],
    # ... other features
})

# Predict credit risk
prediction = model.predict(new_applicant)
print(f"Credit Risk: {'Good' if prediction[0] == 1 else 'Bad'}")

# Get probability scores
proba = model.predict_proba(new_applicant)
print(f"Probability of Good Credit: {proba[0][1]:.2%}")
```

## 📁 Project Structure

```
CodeAlpha_CreditScoringModel/
├── CodeAlpha_CreditScoringModel.ipynb    # Main notebook
├── german.data                           # UCI German Credit Dataset
├── credit_scoring_rf_model.pkl           # Trained Random Forest model
├── CodeAlpha_CreditScoringModel.slides.html  # Presentation slides
├── requirements.txt                      # Python dependencies
└── README.md                             # Project documentation
```

## 📊 Model Performance

### Random Forest Classifier (Best Model)

- **Accuracy:** 75%+
- **ROC-AUC Score:** 0.78
- **Precision:** 0.77
- **Recall:** 0.73
- **F1-Score:** 0.75

### Logistic Regression

- **Accuracy:** 72%
- **ROC-AUC Score:** 0.74
- **Precision:** 0.74
- **Recall:** 0.70
- **F1-Score:** 0.72

## 🔍 Key Insights

**Top 5 Most Important Features:**

1. **Credit History** - Past payment behavior
2. **Account Balance** - Current financial status
3. **Loan Duration** - Length of credit request
4. **Age** - Applicant's age
5. **Employment Status** - Job stability

## 📋 How to Run

1. **Clone or Download** this repository
2. **Open** `CodeAlpha_CreditScoringModel.ipynb` in Jupyter
3. **Run all cells** to reproduce the results
4. **Explore** visualizations and model performance
5. **View slides** in `CodeAlpha_CreditScoringModel.slides.html`

## 📚 Project Overview

### 1. Data Exploration

- Loaded UCI German Credit Data
- Performed exploratory data analysis (EDA)
- Visualized feature distributions
- Identified correlations

### 2. Data Preprocessing

- Handled missing values
- Encoded categorical variables
- Scaled numerical features
- Split into training and testing sets

### 3. Model Training

- Implemented Logistic Regression
- Built Random Forest classifier
- Tuned hyperparameters
- Cross-validation for robustness

### 4. Evaluation

- Calculated precision, recall, F1-score
- Generated ROC curves
- Analyzed confusion matrices
- Interpreted feature importance

### 5. Deployment

- Serialized best model (Random Forest)
- Created presentation slides
- Documented findings and recommendations

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/NewFeature`)
3. Commit your changes (`git commit -m 'Add NewFeature'`)
4. Push to the branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Bhanu Kumar Dev**

- 🎓 3rd Year CSSE @ KIIT University
- 👨‍💻 Code Alpha Intern | Data Science & ML
- 📧 Email: kumarbhanu818@gmail.com
- 💼 LinkedIn: [bhanu-kumar-dev-97b820313](https://www.linkedin.com/in/bhanu-kumar-dev-97b820313)
- 🐙 GitHub: [@bhanukumardev](https://github.com/bhanukumardev)
- 🌐 Portfolio: [bhanukumardev.github.io/bhanu-portfolio](https://bhanukumardev.github.io/bhanu-portfolio/)

## 🌟 Acknowledgments

- **Code Alpha** - For the internship opportunity
- **UCI Machine Learning Repository** - For the dataset
- **Scikit-learn Community** - For excellent documentation

## 📈 Future Enhancements

- [ ] Implement ensemble methods (XGBoost, LightGBM)
- [ ] Add deep learning models
- [ ] Build a web app for real-time predictions
- [ ] Integrate with banking APIs
- [ ] Expand to multi-class credit scoring
- [ ] Deploy to cloud platforms (AWS, Azure)

---

⭐ **Star this repo if you find it helpful!**

*Empowering financial decisions with ML*

**Made with ❤️ by Bhanu Kumar Dev | Code Alpha Internship**