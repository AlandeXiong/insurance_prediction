"""Prediction service"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from src.features.engineering import FeatureEngineer


class PredictionRequest(BaseModel):
    """Request model for prediction API"""
    State: str = Field(..., description="State")
    Coverage: str = Field(..., description="Coverage type")
    Education: str = Field(..., description="Education level")
    EmploymentStatus: str = Field(..., description="Employment status")
    Gender: str = Field(..., description="Gender")
    Income: float = Field(..., description="Income")
    Location_Code: str = Field(..., alias="Location Code", description="Location code")
    Marital_Status: str = Field(..., alias="Marital Status", description="Marital status")
    Monthly_Premium_Auto: float = Field(..., alias="Monthly Premium Auto", description="Monthly premium")
    Months_Since_Last_Claim: int = Field(..., alias="Months Since Last Claim", description="Months since last claim")
    Months_Since_Policy_Inception: int = Field(..., alias="Months Since Policy Inception", description="Months since policy inception")
    Number_of_Open_Complaints: int = Field(..., alias="Number of Open Complaints", description="Number of open complaints")
    Number_of_Policies: int = Field(..., alias="Number of Policies", description="Number of policies")
    Policy_Type: str = Field(..., alias="Policy Type", description="Policy type")
    Policy: str = Field(..., description="Policy")
    Renew_Offer_Type: str = Field(..., alias="Renew Offer Type", description="Renew offer type")
    Sales_Channel: str = Field(..., alias="Sales Channel", description="Sales channel")
    Total_Claim_Amount: float = Field(..., alias="Total Claim Amount", description="Total claim amount")
    Vehicle_Class: str = Field(..., alias="Vehicle Class", description="Vehicle class")
    Vehicle_Size: str = Field(..., alias="Vehicle Size", description="Vehicle size")
    Customer_Lifetime_Value: Optional[float] = Field(None, alias="Customer Lifetime Value", description="Customer lifetime value")
    
    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    """Response model for prediction API"""
    prediction: int = Field(..., description="Prediction (0 or 1)")
    probability: float = Field(..., description="Prediction probability")
    model_name: str = Field(..., description="Model used for prediction")


class Predictor:
    """Prediction service for insurance renewal"""
    
    def __init__(self, models_dir: Path, feature_engineer_path: Path):
        self.models_dir = Path(models_dir)
        self.feature_engineer_path = Path(feature_engineer_path)
        self.models = {}
        self.feature_engineer = None
        self.load_models()
        self.load_feature_engineer()
    
    def load_models(self):
        """Load all trained models"""
        model_files = {
            'xgboost': 'xgboost_model.pkl',
            'catboost': 'catboost_model.pkl',
            'ensemble': 'ensemble_model.pkl'
        }
        
        for name, filename in model_files.items():
            model_path = self.models_dir / filename
            if model_path.exists():
                self.models[name] = joblib.load(model_path)
                print(f"Loaded {name} model")
            else:
                print(f"Warning: {name} model not found at {model_path}")
    
    def load_feature_engineer(self):
        """Load feature engineer"""
        if self.feature_engineer_path.exists():
            self.feature_engineer = FeatureEngineer.load(self.feature_engineer_path)
            print("Loaded feature engineer")
        else:
            raise FileNotFoundError(f"Feature engineer not found at {self.feature_engineer_path}")
    
    def predict(self, request: PredictionRequest, model_name: str = 'ensemble') -> PredictionResponse:
        """Make prediction"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        # Convert request to DataFrame
        data_dict = request.dict(by_alias=True)
        df = pd.DataFrame([data_dict])
        
        # Feature engineering (no y needed for prediction - uses stored mappings)
        df_processed = self.feature_engineer.transform(df, y=None, fit=False)
        
        # Remove target column if present
        if 'Response' in df_processed.columns:
            df_processed = df_processed.drop('Response', axis=1)
        
        # Make prediction
        model = self.models[model_name]
        probability = model.predict_proba(df_processed)[0, 1]
        prediction = 1 if probability >= 0.5 else 0
        
        return PredictionResponse(
            prediction=prediction,
            probability=float(probability),
            model_name=model_name
        )
    
    def predict_batch(self, requests: list, model_name: str = 'ensemble') -> list:
        """Make batch predictions"""
        results = []
        for req in requests:
            result = self.predict(req, model_name)
            results.append(result)
        return results
