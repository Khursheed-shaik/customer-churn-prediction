
# 📊 Customer Churn Prediction System

An end-to-end Machine Learning application that predicts whether a customer is likely to churn based on customer demographics, service subscriptions, contract details, and billing information.

The project covers the complete Machine Learning workflow including data cleaning, exploratory data analysis, feature preprocessing, model comparison, hyperparameter tuning, probability threshold optimization, model serialization, and Streamlit deployment.

## 📌 Project Overview

Customer churn occurs when a customer stops using a company's products or services.

For subscription-based businesses, identifying customers who are likely to churn can help companies take preventive actions such as personalized offers, discounts, customer support, and retention campaigns.

This project uses Machine Learning to predict whether a customer is likely to churn.

### Objective

The system performs binary classification:

- `0` → Customer is likely to stay
- `1` → Customer is likely to churn

The application also provides the customer's predicted churn probability.

---

## 💡 Business Problem

Businesses often discover customer churn only after the customer has already left.

This project provides an early warning system:

```text
Customer Data
     ↓
Data Preprocessing
     ↓
Machine Learning Model
     ↓
Churn Probability
     ↓
Risk Identification
     ↓
Customer Retention Action
```

Businesses can use the predictions to:

- Identify high-risk customers
- Prioritize customer support
- Provide personalized offers
- Improve customer retention
- Reduce potential revenue loss
- Make data-driven decisions

---

## 🧠 Machine Learning Workflow

```text
Customer Dataset
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Feature Preparation
       ↓
Train/Test Split
       ↓
Feature Preprocessing
       ↓
Model Training
       ↓
Model Comparison
       ↓
Hyperparameter Tuning
       ↓
Threshold Optimization
       ↓
Final Model
       ↓
Model Serialization
       ↓
Streamlit Application
       ↓
Cloud Deployment
```

---

## 📂 Dataset

This project uses the **Telco Customer Churn Dataset**.

The dataset contains customer information related to demographics, subscribed services, account information, contract type, payment method, and billing.

### Dataset Source

[Telco Customer Churn Dataset - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

### Dataset Information

| Property | Value |
|---|---|
| Dataset | Telco Customer Churn |
| Records | 7,043 |
| Columns | 21 |
| Target Variable | `Churn` |
| Problem Type | Binary Classification |

---

## 📋 Dataset Features

The main features used by the model include:

- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`
- `tenure`
- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`
- `MonthlyCharges`
- `TotalCharges`

---

## 🧹 Data Preprocessing

### 1. Handling TotalCharges

The `TotalCharges` column contained values stored as strings, so it was converted into numerical format.

```python
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
```

Missing values were handled using the median.

### 2. Removing Customer ID

The `customerID` column was removed because it is only an identifier and does not provide useful predictive information.

### 3. Target Encoding

The target variable was converted into binary values:

```text
Yes → 1
No  → 0
```

### 4. Numerical Feature Scaling

Numerical features were standardized using:

```text
StandardScaler
```

### 5. Categorical Feature Encoding

Categorical features were converted into numerical representations using:

```text
OneHotEncoder(handle_unknown="ignore")
```

### 6. Preprocessing Pipeline

The preprocessing steps were combined with the Machine Learning model using a Scikit-learn pipeline.

This ensures that the same preprocessing is automatically applied when making predictions on new customers.

---

## 📊 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand customer behavior and identify relationships with churn.

The analysis included:

- Churn distribution
- Contract type vs churn
- Customer tenure vs churn
- Monthly charges vs churn
- Service subscriptions
- Payment methods
- Customer demographics

The analysis showed that factors such as tenure, contract type, monthly charges, internet service, and payment method can be associated with customer churn.

---

## 🤖 Machine Learning Models

Three classification algorithms were trained and evaluated:

### 1. Logistic Regression

Used as an interpretable baseline model for binary classification.

### 2. Random Forest

Used as an ensemble tree-based model capable of capturing non-linear relationships.

### 3. XGBoost

Used as a gradient boosting model to compare performance against traditional classification approaches.

---

## 📈 Model Comparison

The baseline model results were:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.55% | 65.72% | 55.88% | 60.40% | 84.19% |
| Random Forest | 78.21% | 61.75% | 47.06% | 53.41% | 82.17% |
| XGBoost | 79.56% | 64.33% | 51.60% | 57.27% | 83.63% |

Logistic Regression achieved the strongest baseline performance and was selected for further optimization.

---

## ⚙️ Hyperparameter Tuning

Logistic Regression was optimized using `GridSearchCV`.

The following parameters were evaluated:

```python
param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__class_weight": [None, "balanced"]
}
```

A 5-fold cross-validation strategy was used.

The optimization metric was **F1-Score**, since both precision and recall are important for churn prediction.

---

## 🎯 Probability Threshold Optimization

The default classification threshold of a binary classification model is generally `0.50`.

However, for churn prediction, the threshold can be adjusted depending on the business objective.

Thresholds between `0.30` and `0.70` were evaluated.

The final threshold selected was:

```text
0.55
```

This provided a better balance between precision and recall for the project.

---

## 🏆 Final Model Performance

The final Logistic Regression model using a classification threshold of `0.55` achieved:

| Metric | Score |
|---|---:|
| Accuracy | **75.30%** |
| Precision | **52.43%** |
| Recall | **75.13%** |
| F1-Score | **61.76%** |
| ROC-AUC | **84.13%** |

---

## 📊 Confusion Matrix

The final model produced:

```text
                 Predicted
                 Stay   Churn

Actual Stay       780     255
Actual Churn       93     281
```

### Results

- True Negatives: `780`
- False Positives: `255`
- False Negatives: `93`
- True Positives: `281`

---

## 🎯 Why Recall Matters

For customer churn prediction, identifying customers who are actually going to leave is important.

A false negative means:

```text
Actual Churn
     ↓
Model predicts Stay
```

This represents a missed opportunity for customer retention.

The final model achieved:

```text
Recall = 75.13%
```

Therefore, the model identifies approximately 75% of the actual churn customers in the test dataset.

Businesses can use these predictions to prioritize customers for retention campaigns.

---

## 🖥️ Streamlit Application

The trained Machine Learning model was integrated into a Streamlit web application.

The application allows users to enter customer information and receive a churn prediction.

### Application Workflow

```text
Customer Information
        ↓
Preprocessing Pipeline
        ↓
Logistic Regression Model
        ↓
Churn Probability
        ↓
Threshold = 0.55
        ↓
Churn Prediction
```

The application displays the predicted churn probability and churn classification.

---

## 🌐 Deployment

The application is deployed using **Streamlit Community Cloud**.

### Live Application

👉 [**Customer Churn Prediction App**](http://tytugbyqvaxowwnjxs92bh.streamlit.app/)

---

## 🛠️ Technologies Used

### Programming

- Python

### Machine Learning

- Scikit-learn
- Logistic Regression
- Random Forest
- XGBoost

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Streamlit

### Model Persistence

- Joblib

### Development Tools

- Google Colab
- VS Code
- Jupyter Notebook

### Version Control

- Git
- GitHub

### Deployment

- Streamlit Community Cloud

---

## 📁 Project Structure

```text
customer-churn-prediction/
│
├── app.py
├── churn_model.pkl
├── requirements.txt
└── README.md
```

### `app.py`

Contains the Streamlit application and prediction logic.

### `churn_model.pkl`

Contains the trained Machine Learning pipeline and optimized prediction threshold.

### `requirements.txt`

Contains the Python dependencies required to run the application.

### `README.md`

Contains the complete project documentation.

---

## ⚡ Installation and Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Khursheed-shaik/customer-churn-prediction.git
```

### 2. Navigate to the Project

```bash
cd customer-churn-prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### macOS/Linux

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📦 Requirements

The project uses:

```text
streamlit
pandas
numpy
scikit-learn==1.6.1
joblib
xgboost
```

The Scikit-learn version is pinned to maintain compatibility with the serialized Machine Learning model.

---

## 🔑 Key Features

- ✅ Customer churn prediction
- ✅ End-to-end Machine Learning pipeline
- ✅ Data cleaning
- ✅ Exploratory Data Analysis
- ✅ Numerical feature scaling
- ✅ Categorical feature encoding
- ✅ Logistic Regression
- ✅ Random Forest
- ✅ XGBoost
- ✅ Model comparison
- ✅ Hyperparameter tuning
- ✅ 5-fold cross-validation
- ✅ Probability threshold optimization
- ✅ Churn probability prediction
- ✅ Saved Machine Learning pipeline
- ✅ Streamlit web application
- ✅ Cloud deployment
- ✅ GitHub repository

---

## 💼 Business Impact

The system can help subscription-based businesses identify customers who have a high probability of churning.

Potential applications include:

### Customer Retention

Identify high-risk customers before they leave.

### Personalized Offers

Provide targeted discounts or incentives to customers with high churn probability.

### Customer Support

Prioritize support for customers who are more likely to leave.

### Revenue Protection

Early identification of churn risk can help businesses reduce potential customer and revenue loss.

### Data-Driven Decision Making

The model provides a quantitative churn probability that can support customer retention strategies.

---

## 📚 Skills Demonstrated

This project demonstrates practical knowledge of:

- Python
- Pandas
- NumPy
- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Categorical Encoding
- Feature Scaling
- Binary Classification
- Logistic Regression
- Random Forest
- XGBoost
- Model Evaluation
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Cross-Validation
- GridSearchCV
- Hyperparameter Tuning
- Probability Threshold Optimization
- Scikit-learn Pipelines
- Joblib
- Streamlit
- Git
- GitHub
- Cloud Deployment

---

## 🔮 Future Improvements

Possible improvements include:

- [ ] Bulk CSV prediction
- [ ] Interactive churn analytics dashboard
- [ ] SHAP-based model explainability
- [ ] Feature importance visualization
- [ ] Customer risk scoring
- [ ] Automated retention recommendations
- [ ] Customer segmentation
- [ ] Model monitoring
- [ ] Automated model retraining
- [ ] Database integration
- [ ] Authentication
- [ ] REST API for predictions

---

## 🎓 Learning Outcomes

Through this project, I gained practical experience in building and deploying a complete Machine Learning application.

Key learning outcomes include:

1. Building an end-to-end Machine Learning workflow.
2. Cleaning and preprocessing real-world tabular data.
3. Handling categorical and numerical features.
4. Building reusable Scikit-learn pipelines.
5. Comparing multiple classification algorithms.
6. Performing hyperparameter tuning using GridSearchCV.
7. Evaluating classification models using multiple metrics.
8. Understanding the precision-recall trade-off.
9. Optimizing probability classification thresholds.
10. Saving and loading trained Machine Learning models.
11. Building interactive ML applications using Streamlit.
12. Deploying Machine Learning applications to the cloud.
13. Managing projects using Git and GitHub.

---

## 👩‍💻 Author

### Khursheed Shaik

**B.Tech – Computer Science and Engineering (AI & ML)**  
Vishnu Institute of Technology

### GitHub

👉 [Khursheed Shaik](https://github.com/Khursheed-shaik)

### Project Repository

👉 [Customer Churn Prediction](https://github.com/Khursheed-shaik/customer-churn-prediction)

### Live Application

👉 [Customer Churn Prediction App](http://tytugbyqvaxowwnjxs92bh.streamlit.app/)

---

## ⭐ Acknowledgements

- [Kaggle](https://www.kaggle.com/) - Telco Customer Churn Dataset
- [Scikit-learn](https://scikit-learn.org/) - Machine Learning framework
- [Streamlit](https://streamlit.io/) - Application and deployment framework
- [XGBoost](https://xgboost.readthedocs.io/) - Gradient boosting
- Python Open-Source Community

---

## 📜 License

This project is developed for educational, learning, and portfolio purposes.
