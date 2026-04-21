import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import pickle
import os

# Load feature dataset
print("Loading features...")
df = pd.read_csv('data/features.csv')
X = df.drop('priority', axis=1)
y = df['priority']

# Split into train and test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define models to compare
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# Cross-validation settings
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}
best_model = None
best_f1 = 0

print("\n" + "="*60)
print("Model Comparison")
print("="*60)

for name, model in models.items():
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    # Train on full training set
    model.fit(X_train, y_train)
    # Predict on test set
    y_pred = model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred)
    test_acc = accuracy_score(y_test, y_pred)
    
    results[name] = {
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std(),
        'test_f1': test_f1,
        'test_acc': test_acc
    }
    print(f"\n{name}:")
    print(f"  CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test F1: {test_f1:.4f}")
    print(f"  Test Acc: {test_acc:.4f}")
    
    # Select best model by test F1
    if test_f1 > best_f1:
        best_f1 = test_f1
        best_model = model

print("\n" + "="*60)
print(f"Best model: {best_model.__class__.__name__} with Test F1 = {best_f1:.4f}")
print("="*60)

# Save the best model
os.makedirs('backend/models', exist_ok=True)
with open('backend/models/ml_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("Best model saved to backend/models/ml_model.pkl")

# Also save feature columns (same as before)
with open('backend/models/feature_columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)
print("Feature columns saved.")