# Telecom Customer Churn Prediction

## Project Overview
This project builds a state-of-sart machine learning model to predict whether a telecom customer will churn based on their demographics, account information, and service usage. The project includes a full pipeline covering data extraction, cleaning, exploratory data analysis (EDA), model training, evaluation, and an interactive web interface using Streamlit.

## Dataset Description
The model uses the **Telco Customer Churn dataset**, initially from IBM/Kaggle. It contains `7043` rows and `21` columns, capturing details about customer demographic, services they have signed up for, and their account information (like tenure, monthly charges, payment method).

**Target Variable**:
* `Churn`: Whether the customer left the company (1 = Yes, 0 = No)

## Technologies Used
* **Python**: Core programming language
* **Pandas & NumPy**: Data loading, cleaning, and manipulation
* **Scikit-learn**: Machine learning model training and evaluation
* **Matplotlib & Seaborn**: Exploratory Data Analysis (EDA) visualizations
* **Streamlit**: Web server and interactive UI dashboard for deployment
* **Joblib**: Model serialization and saving

## Methodology

1. **Data Loading & Cleaning**:
   - The data is loaded using `pandas`.
   - `TotalCharges` is converted to numeric and empty values are handled.
   - The categorical target variable `Churn` is mapped to binaries (1/0).
   - Unnecessary identifier `customerID` is removed.

2. **Exploratory Data Analysis (EDA)**:
   - Several plots are generated, including the distribution of churn rate, and bivariate analysis like churn vs contract type, tenure, and monthly charges.
   - A correlation heatmap is provided for numerical variables.

3. **Feature Engineering**:
   - Categorical variables are encoded using Scikit-Learn's `OneHotEncoder`.
   - Continuous attributes are standardized with `StandardScaler`.

4. **Model Training & Comparison**:
   - Models evaluated: Logistic Regression, Decision Tree, Random Forest.
   - The pipeline approach ensures clean preprocessing application across Train and Test datasets without data leakage.

5. **Model Evaluation & Selection**:
   - Models are evaluated using Accuracy, Precision, Recall, and ROC-AUC. 
   - The **Logistic Regression** model proved to be the most performant with an ROC-AUC score of **0.8361**.
   - The model is saved as `models/churn_model.pkl` using `joblib`.

## Results
The selected Logistic Regression model provides solid predictive capability:
- **Accuracy**: 80.53%
- **ROC-AUC**: 0.8361

## Instructions to Run Locally

### Prerequisites
Make sure you have Python 3.8+ installed.

### 1. Clone the repository and navigate to the root directory
```bash
git clone <repository_url>
cd customer-churn-prediction
```

### 2. Install Dependencies
Install all required libraries specified in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Run the Data Pipeline & Model Training
Execute the training script. This script will load the dataset (make sure `WA_Fn-UseC_-Telco-Customer-Churn.csv` is in `data/`), clean it, generate EDA plots in the `app/` folder, train multiple models, evaluate them, and save the best model to the `models/` directory.
```bash
python src/train_model.py
```

### 4. Run the Streamlit Application
Launch the interactive prediction app and dashboard using Streamlit:
```bash
streamlit run app/main.py
```

The app will become available at `http://localhost:8501`. You can enter any customer's details and get a live churn prediction.
