"""Test script for API"""
import requests
import json

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{API_URL}/health")
    print("Health Check:", response.json())
    return response.status_code == 200

def test_predict():
    """Test prediction endpoint"""
    sample_data = {
        "State": "California",
        "Coverage": "Premium",
        "Education": "Bachelor",
        "EmploymentStatus": "Employed",
        "Gender": "M",
        "Income": 50000.0,
        "Location Code": "Suburban",
        "Marital Status": "Married",
        "Monthly Premium Auto": 100.0,
        "Months Since Last Claim": 12,
        "Months Since Policy Inception": 24,
        "Number of Open Complaints": 0,
        "Number of Policies": 1,
        "Policy Type": "Personal Auto",
        "Policy": "Personal L3",
        "Renew Offer Type": "Offer1",
        "Sales Channel": "Agent",
        "Total Claim Amount": 500.0,
        "Vehicle Class": "Four-Door Car",
        "Vehicle Size": "Medsize"
    }
    
    response = requests.post(
        f"{API_URL}/predict",
        json=sample_data,
        params={"model": "ensemble"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print("\nPrediction Result:")
        print(json.dumps(result, indent=2))
        return True
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return False

def test_models():
    """Test models endpoint"""
    response = requests.get(f"{API_URL}/models")
    print("\nAvailable Models:", response.json())
    return response.status_code == 200

if __name__ == "__main__":
    print("Testing Insurance Renewal Prediction API\n")
    print("="*50)
    
    # Test health
    print("\n1. Testing health endpoint...")
    test_health()
    
    # Test models
    print("\n2. Testing models endpoint...")
    test_models()
    
    # Test prediction
    print("\n3. Testing prediction endpoint...")
    test_predict()
    
    print("\n" + "="*50)
    print("Testing complete!")
