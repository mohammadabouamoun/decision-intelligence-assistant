import pickle
import pandas as pd
import time
import numpy as np
from .helpers import extract_features_for_ml

ml_model = None
feature_columns = None

def init_ml():
    global ml_model, feature_columns
    with open('backend/models/ml_model.pkl', 'rb') as f:
        ml_model = pickle.load(f)
    with open('backend/models/feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    print("ML model initialized.")

def predict_priority(text: str):
    start = time.time()
    features = extract_features_for_ml(text)
    df = pd.DataFrame([features])[feature_columns]
    pred_proba = ml_model.predict_proba(df)[0]
    confidence = float(np.max(pred_proba))   # convert numpy float to Python float
    label = ml_model.classes_[pred_proba.argmax()]
    latency_ms = (time.time() - start) * 1000
    return label, confidence, latency_ms