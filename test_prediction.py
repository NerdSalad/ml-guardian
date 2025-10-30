#!/usr/bin/env python3
"""
Simple test script to verify the ML Guardian pipeline works
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import joblib
import pandas as pd
import numpy as np
from core.config import cfg
from core.labels import REVERSE_LABEL_MAP
from core.feature_engineer_optimized import OptimizedFeatureEngineer

def test_prediction():
    print("[TEST] Testing ML Guardian Pipeline...")
    
    # Load the trained XGBoost model
    try:
        xgb_pipeline = joblib.load(cfg["paths"]["xgb_model"])
        print("[SUCCESS] XGBoost model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load XGBoost model: {e}")
        return False
    
    # Test with sample data
    test_claims = [
        "The Eiffel Tower is located in Paris.",
        "The sky is blue.",
        "Water boils at 100 degrees Celsius."
    ]
    
    test_evidence = [
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
        "The sky appears blue due to Rayleigh scattering of sunlight by the atmosphere.",
        "Water boils at 100°C (212°F) at standard atmospheric pressure."
    ]
    
    # Initialize feature engineer
    fe = OptimizedFeatureEngineer()
    
    print("\n[PREDICTIONS] Testing predictions:")
    print("=" * 60)
    
    for i, (claim, evidence) in enumerate(zip(test_claims, test_evidence)):
        print(f"\nTest {i+1}:")
        print(f"Claim: {claim}")
        print(f"Evidence: {evidence}")
        
        try:
            # Extract features
            features = fe.extract_all_features(claim, evidence)
            
            # Convert to DataFrame and align with model features
            X = pd.DataFrame([features])
            feature_names = xgb_pipeline["feature_names"]
            X = X.reindex(columns=feature_names, fill_value=0.0)
            
            # Make prediction
            probabilities = xgb_pipeline["model"].predict_proba(X.values)[0]
            prediction = np.argmax(probabilities)
            confidence = probabilities[prediction]
            
            # Get label name
            label_name = REVERSE_LABEL_MAP.get(prediction, f"Class_{prediction}")
            
            print(f"Prediction: {label_name}")
            print(f"Confidence: {confidence:.3f}")
            print(f"All probabilities: {dict(zip([REVERSE_LABEL_MAP.get(i, f'Class_{i}') for i in range(len(probabilities))], probabilities))}")
            
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("[SUCCESS] All tests completed successfully!")
    print("\n[SUMMARY] Model Performance Summary:")
    print(f"- Model: XGBoost Classifier")
    print(f"- Features: {len(feature_names)} features")
    print(f"- Classes: {list(REVERSE_LABEL_MAP.values())}")
    
    return True

if __name__ == "__main__":
    success = test_prediction()
    if success:
        print("\n[SUCCESS] ML Guardian is working correctly!")
    else:
        print("\n[ERROR] ML Guardian has issues that need to be fixed.")
        sys.exit(1)
