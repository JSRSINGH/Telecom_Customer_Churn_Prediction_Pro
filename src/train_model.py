import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
import joblib
import os

# Set plotting style
plt.style.use('ggplot')
sns.set_palette('muted')

def load_data(filepath):
    """Loads the dataset and prints basic info."""
    print("Loading data...")
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print("\nDataset Info:")
    df.info()
    return df

def clean_data(df):
    """Cleans the dataset according to requirements."""
    print("\nCleaning data...")
    # Handle TotalCharges: Convert to numeric, forcing errors to NaN, then fill or drop
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Drop rows with NaN in TotalCharges (only ~11 rows)
    df.dropna(subset=['TotalCharges'], inplace=True)
    
    # Convert Churn to 1/0
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Drop customerID as it's not useful for prediction
    df.drop('customerID', axis=1, inplace=True)
    
    return df

def perform_eda(df, output_dir):
    """Generates and saves EDA visualizations."""
    print("\nPerforming EDA...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Churn Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Churn')
    plt.title('Churn Distribution')
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.savefig(os.path.join(output_dir, 'churn_distribution.png'))
    plt.close()
    
    # 2. Churn vs Contract type
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Contract', hue='Churn')
    plt.title('Churn by Contract Type')
    plt.savefig(os.path.join(output_dir, 'churn_vs_contract.png'))
    plt.close()
    
    # 3. Churn vs Tenure (Boxplot or KDE)
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='Churn', y='tenure')
    plt.title('Tenure Distribution by Churn')
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.savefig(os.path.join(output_dir, 'churn_vs_tenure.png'))
    plt.close()
    
    # 4. Churn vs Monthly charges
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='Churn', y='MonthlyCharges')
    plt.title('Monthly Charges Distribution by Churn')
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.savefig(os.path.join(output_dir, 'churn_vs_monthly_charges.png'))
    plt.close()
    
    # 5. Correlation Heatmap (only numeric features)
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()

def build_model_pipeline(categorical_cols, numerical_cols, model):
    """Builds a scikit-learn pipeline with preprocessing and the given model."""
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
    return clf

def evaluate_model(y_true, y_pred, y_prob, model_name):
    """Evaluates and prints model metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"--- {model_name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("-" * 25)
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'roc_auc': roc_auc, 'cm': cm}

def get_feature_names(column_transformer, categorical_cols, numerical_cols):
    """Extracts feature names after one-hot encoding."""
    cat_encoder = column_transformer.named_transformers_['cat'].named_steps['onehot']
    cat_features = cat_encoder.get_feature_names_out(categorical_cols)
    return list(numerical_cols) + list(cat_features)

def plot_feature_importance(model, feature_names, output_dir, model_name):
    """Plots and saves feature importance."""
    if model_name == 'Logistic Regression':
        importances = np.abs(model.coef_[0])
    else:
        importances = model.feature_importances_
        
    indices = np.argsort(importances)[-15:] # Top 15 features
    
    plt.figure(figsize=(10, 6))
    plt.title('Top 15 Feature Importances')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()

def main():
    data_path = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    eda_output_dir = 'app'
    model_output_path = 'models/churn_model.pkl'
    
    # 1. Load Data
    df = load_data(data_path)
    
    # 2. Clean Data
    df = clean_data(df)
    
    # 3. EDA
    perform_eda(df, eda_output_dir)
    
    # 4. Feature Engineering & Split
    print("\nPreparing data for modeling...")
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # 5. Model Training & Evaluation
    print("\nTraining models...")
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    best_model_name = None
    best_roc_auc = 0
    best_pipeline = None
    
    for name, model in models.items():
        pipeline = build_model_pipeline(categorical_cols, numerical_cols, model)
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        metrics = evaluate_model(y_test, y_pred, y_prob, name)
        
        if metrics['roc_auc'] > best_roc_auc:
            best_roc_auc = metrics['roc_auc']
            best_model_name = name
            best_pipeline = pipeline
            
    print(f"\nBest Model: {best_model_name} with ROC-AUC = {best_roc_auc:.4f}")
    
    # 6. Feature Importance Extract
    classifier = best_pipeline.named_steps['classifier']
    preprocessor = best_pipeline.named_steps['preprocessor']
    feature_names = get_feature_names(preprocessor, categorical_cols, numerical_cols)
    plot_feature_importance(classifier, feature_names, eda_output_dir, best_model_name)
    print("Feature importance plot saved.")
    
    # 7. Save Model
    print(f"\nSaving best model to {model_output_path}...")
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(best_pipeline, model_output_path)
    print("Model saved successfully.")

if __name__ == '__main__':
    main()
