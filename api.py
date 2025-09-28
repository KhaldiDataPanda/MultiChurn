from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import json
import io
from typing import Dict, List
import os
from contextlib import asynccontextmanager
from train_models import BankChurnModel, TelecomChurnModel

app = FastAPI(title="Multi-Churn Prediction API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store models - These will be populated by the lifespan event
bank_model = None
telecom_model = None
model_info = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    global bank_model, telecom_model, model_info
    print("Loading models...")
    try:
        bank_model = joblib.load('models/bank_churn_model.pkl')
        telecom_model = joblib.load('models/telecom_churn_model.pkl')
        with open('models/model_info.json', 'r') as f:
            model_info = json.load(f)
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
    yield
    # Clean up models on shutdown
    print("Clearing models...")
    bank_model = None
    telecom_model = None
    model_info = None

app = FastAPI(title="Multi-Churn Prediction API", version="1.0.0", lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Multi-Churn Prediction API is running!"}

@app.get("/model-info/{model_type}")
async def get_model_info(model_type: str):
    """Get information about required columns for a specific model"""
    if model_info is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    if model_type not in model_info:
        raise HTTPException(status_code=404, detail=f"Model type '{model_type}' not found")
    
    return model_info[model_type]

@app.post("/validate-data/{model_type}")
async def validate_data(model_type: str, file: UploadFile = File(...)):
    """Validate uploaded data against model requirements"""
    if model_info is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    if model_type not in model_info:
        raise HTTPException(status_code=404, detail=f"Model type '{model_type}' not found")
    
    try:
        # Read the uploaded file
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        required_cols = model_info[model_type]['required_columns']
        uploaded_cols = df.columns.tolist()
        
        # Check for missing columns
        missing_cols = [col for col in required_cols if col not in uploaded_cols]
        
        # Check for extra columns
        extra_cols = [col for col in uploaded_cols if col not in required_cols]
        
        # Check for missing values in required columns
        available_required_cols = [col for col in required_cols if col in uploaded_cols]
        missing_values = {}
        
        for col in available_required_cols:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_values[col] = int(missing_count)
        
        # Check data types
        type_issues = {}
        for col in available_required_cols:
            if col in ['tenure', 'MonthlyCharges', 'TotalCharges', 'Age', 'CreditScore', 'Balance', 'EstimatedSalary', 'NumOfProducts', 'SeniorCitizen', 'IsActiveMember']:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    # Try to convert to numeric
                    try:
                        pd.to_numeric(df[col], errors='raise')
                    except:
                        type_issues[col] = f"Expected numeric, got {df[col].dtype}"
        
        validation_result = {
            "is_valid": len(missing_cols) == 0 and len(missing_values) == 0 and len(type_issues) == 0,
            "data_shape": df.shape,
            "missing_columns": missing_cols,
            "extra_columns": extra_cols,
            "missing_values": missing_values,
            "type_issues": type_issues,
            "required_columns": required_cols,
            "uploaded_columns": uploaded_cols
        }
        
        return validation_result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

@app.post("/predict/{model_type}")
async def predict_churn(model_type: str, file: UploadFile = File(...)):
    """Make predictions using the specified model"""
    if model_type not in ["bank_churn", "telecom_churn"]:
        raise HTTPException(status_code=404, detail=f"Model type '{model_type}' not found")
    
    if (model_type == "bank_churn" and bank_model is None) or \
       (model_type == "telecom_churn" and telecom_model is None):
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read the uploaded file
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Select the appropriate model
        model = bank_model if model_type == "bank_churn" else telecom_model
        
        # Make predictions
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)
        
        # Create result dataframe
        result_df = df.copy()
        result_df['Predicted_Churn'] = predictions
        result_df['Churn_Probability'] = probabilities[:, 1]  # Probability of churn (class 1)
        
        # Convert to records for JSON response
        predictions_list = result_df.to_dict('records')
        
        # Calculate summary statistics
        total_customers = len(predictions)
        churn_count = int(np.sum(predictions))
        no_churn_count = total_customers - churn_count
        churn_percentage = (churn_count / total_customers * 100) if total_customers > 0 else 0
        
        summary = {
            "total_customers": total_customers,
            "predicted_churn": churn_count,
            "predicted_no_churn": no_churn_count,
            "churn_percentage": round(churn_percentage, 2)
        }
        
        return {
            "predictions": predictions_list,
            "summary": summary,
            "model_type": model_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error making predictions: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "bank_model_loaded": bank_model is not None,
        "telecom_model_loaded": telecom_model is not None,
        "model_info_loaded": model_info is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)