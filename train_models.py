import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from category_encoders import TargetEncoder
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class BankChurnModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.target_encoder = TargetEncoder()
        self.scaler = StandardScaler()
        self.required_columns = [
            'Age', 'IsActiveMember', 'Gender', 'NumOfProducts', 
            'CreditScore', 'Balance', 'Geography', 'Tenure', 'EstimatedSalary'
        ]
        self.categorical_cols = ['Gender', 'Geography', 'IsActiveMember']
        self.numerical_cols = ['Age', 'NumOfProducts', 'CreditScore', 'Balance', 'Tenure', 'EstimatedSalary']
        
    def preprocess_data(self, df, is_training=True):
        # Create a copy to avoid modifying original data
        df = df.copy()
        
        # Drop unnecessary columns if they exist
        cols_to_drop = ['CustomerId', 'Surname', 'RowNumber']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        
        # Handle missing values
        df = df.dropna()
        
        # Feature engineering
        df['CAR'] = df['CreditScore'] / df['Age']
        df['CSR'] = df['CreditScore'] / df['EstimatedSalary']
        df['TPA'] = df['Tenure'] / df['Age']
        
        # Handle infinite values
        df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
        df.fillna(0, inplace=True)
        
        # Update required columns to include engineered features
        feature_cols = self.required_columns + ['CAR', 'CSR', 'TPA']
        
        # Target encoding for categorical columns
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if is_training:
            if 'Exited' in df.columns:
                df[cat_cols] = self.target_encoder.fit_transform(df[cat_cols], df['Exited'])
            else:
                raise ValueError("Target column 'Exited' not found in training data")
        else:
            df[cat_cols] = self.target_encoder.transform(df[cat_cols])
        
        # Select features for model
        X = df[feature_cols]
        
        # Scale numerical features
        numerical_cols_to_scale = [col for col in feature_cols if col not in self.categorical_cols]
        X_numerical = X[numerical_cols_to_scale]
        X_categorical = X[self.categorical_cols] if any(col in X.columns for col in self.categorical_cols) else pd.DataFrame()
        
        if is_training:
            X_numerical_scaled = self.scaler.fit_transform(X_numerical)
        else:
            X_numerical_scaled = self.scaler.transform(X_numerical)
        
        X_numerical_scaled_df = pd.DataFrame(X_numerical_scaled, columns=numerical_cols_to_scale, index=X_numerical.index)
        
        if not X_categorical.empty:
            X_final = pd.concat([X_numerical_scaled_df.reset_index(drop=True), X_categorical.reset_index(drop=True)], axis=1)
        else:
            X_final = X_numerical_scaled_df
            
        return X_final
    
    def train(self, data_path):
        # Load data
        df = pd.read_csv(data_path)
        
        # Preprocess data
        X = self.preprocess_data(df, is_training=True)
        y = df['Exited']
        
        # Split data
        train_x, test_x, train_y, test_y = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Handle class imbalance with upsampling
        train_data = pd.concat([train_x, train_y], axis=1)
        df_majority = train_data[train_data['Exited'] == 0]
        df_minority = train_data[train_data['Exited'] == 1]
        df_minority_upsampled = resample(
            df_minority, replace=True, n_samples=len(df_majority), random_state=42
        )
        
        train_balanced = pd.concat([df_majority, df_minority_upsampled])
        train_x_balanced = train_balanced.drop('Exited', axis=1)
        train_y_balanced = train_balanced['Exited']
        
        # Train model
        self.model.fit(train_x_balanced, train_y_balanced)
        
        # Evaluate on test set
        test_score = self.model.score(test_x, test_y)
        print(f"Bank Churn Model Test Accuracy: {test_score:.3f}")
        
        return self.model
    
    def predict(self, X):
        X_processed = self.preprocess_data(X, is_training=False)
        return self.model.predict(X_processed)
    
    def predict_proba(self, X):
        X_processed = self.preprocess_data(X, is_training=False)
        return self.model.predict_proba(X_processed)


class TelecomChurnModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.target_encoder = TargetEncoder()
        self.scalers = {}
        self.required_columns = [
            'tenure', 'MonthlyCharges', 'TotalCharges', 'gender', 'SeniorCitizen',
            'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
        ]
        
    def preprocess_data(self, df, is_training=True):
        # Create a copy to avoid modifying original data
        df = df.copy()
        
        # Drop customerID if it exists
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        # Convert TotalCharges to numeric (handle empty strings)
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'].fillna(0, inplace=True)
        
        # Handle Churn column if it exists (for training)
        if 'Churn' in df.columns and is_training:
            df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
        
        # Target encoding for categorical columns
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        if 'Churn' in cat_cols:
            cat_cols.remove('Churn')
            
        if is_training and 'Churn' in df.columns:
            df[cat_cols] = self.target_encoder.fit_transform(df[cat_cols], df['Churn'])
        elif not is_training:
            df[cat_cols] = self.target_encoder.transform(df[cat_cols])
        
        # Scale numerical columns
        col2scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
        for col in col2scale:
            if col in df.columns:
                if is_training:
                    scaler = StandardScaler()
                    df[col] = scaler.fit_transform(df[[col]])
                    self.scalers[col] = scaler
                else:
                    if col in self.scalers:
                        df[col] = self.scalers[col].transform(df[[col]])
        
        # Select features (exclude MultipleLines as per notebook)
        feature_cols = [x for x in self.required_columns if x != 'MultipleLines' and x in df.columns]
        X = df[feature_cols]
        
        return X
    
    def train(self, data_path):
        # Load data
        df = pd.read_csv(data_path)
        
        # Preprocess data
        X = self.preprocess_data(df, is_training=True)
        y = df['Churn'].map({'No': 0, 'Yes': 1}) if df['Churn'].dtype == 'object' else df['Churn']
        
        # Split data
        train_x, test_x, train_y, test_y = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model.fit(train_x, train_y)
        
        # Evaluate on test set
        test_score = self.model.score(test_x, test_y)
        print(f"Telecom Churn Model Test Accuracy: {test_score:.3f}")
        
        return self.model
    
    def predict(self, X):
        X_processed = self.preprocess_data(X, is_training=False)
        return self.model.predict(X_processed)
    
    def predict_proba(self, X):
        X_processed = self.preprocess_data(X, is_training=False)
        return self.model.predict_proba(X_processed)


def train_and_save_models():
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    print("Training Bank Churn Model...")
    bank_model = BankChurnModel()
    try:
        bank_model.train('data/Banking.csv')
        # Save the entire model object
        joblib.dump(bank_model, 'models/bank_churn_model.pkl')
        print("Bank Churn Model saved successfully!")
    except Exception as e:
        print(f"Error training Bank Churn Model: {e}")
    
    print("\nTraining Telecom Churn Model...")
    telecom_model = TelecomChurnModel()
    try:
        telecom_model.train('data/Telecom.csv')
        # Save the entire model object
        joblib.dump(telecom_model, 'models/telecom_churn_model.pkl')
        print("Telecom Churn Model saved successfully!")
    except Exception as e:
        print(f"Error training Telecom Churn Model: {e}")
    
    # Save model information
    model_info = {
        'bank_churn': {
            'required_columns': bank_model.required_columns,
            'target_column': 'Exited',
            'model_type': 'RandomForestClassifier'
        },
        'telecom_churn': {
            'required_columns': telecom_model.required_columns,
            'target_column': 'Churn',
            'model_type': 'RandomForestClassifier'
        }
    }
    
    import json
    with open('models/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("\nAll models trained and saved successfully!")
    print("Model files saved in 'models' directory:")
    print("- bank_churn_model.pkl")
    print("- telecom_churn_model.pkl")
    print("- model_info.json")


if __name__ == "__main__":
    train_and_save_models()