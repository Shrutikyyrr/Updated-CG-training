"""
Day 8 - K-Fold Cross Validation, Bias-Variance & Encoding
==========================================================
Scenarios:
  1. Student Grade Prediction (Ridge Regression + K-Fold CV)
  2. Patient Recovery Time Prediction (Linear Regression + 5-Fold CV)
  3. Student Exam Performance Prediction (Linear Regression + 5-Fold CV)
  4. Bias vs Variance - Student Exam Scores (Linear vs Polynomial)
  5. Package Delivery System (Label Encoding)
  6. Restaurant Ordering System (One-Hot Encoding)
  7. Employee Training & Satisfaction Survey (Ordinal Encoding)
  8. Patient Health Monitoring (Label + One-Hot Encoding)
  9. Online Food Delivery App (One-Hot Encoding)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression

# ─────────────────────────────────────────────────────────────
# SCENARIO 1: Student Grade Prediction
# ─────────────────────────────────────────────────────────────
# A school wants to predict student grades based on 5 factors
# like attendance, homework scores, and test results.
# The dataset has 1000 students. To make sure the model works
# well on unseen students (not just the ones it trained on),
# we use K-Fold Cross Validation with 5 splits.
# We use Ridge Regression to avoid overfitting.
# Goal: Check if the model is consistent across all 5 folds.
print("=" * 60)
print("SCENARIO 1: Student Grade Prediction")
print("Ridge Regression + K-Fold Cross Validation")
print("=" * 60)

X, y = make_regression(n_samples=1000, n_features=5, noise=15, random_state=42)
model = Ridge()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
print(f"R2 scores for each fold: {scores.round(3)}")
print(f"Mean R2: {scores.mean().round(3)}")
print(f"Std Dev: {scores.std().round(3)}")
print("Model is stable across folds." if scores.std() < 0.05 else "Model varies across folds.")

# ─────────────────────────────────────────────────────────────
# SCENARIO 2: Patient Recovery Time Prediction
# ─────────────────────────────────────────────────────────────
# A hospital wants to predict how many days a patient will take
# to recover after surgery. The dataset has 1000 patients with
# 5 medical features (age, blood pressure, medication dose, etc.).
# Doctors need a reliable model, so we test it using 5-Fold CV
# to make sure it performs consistently on different patient groups.
# We use Linear Regression as the base model.
print("\n" + "=" * 60)
print("SCENARIO 2: Patient Recovery Time Prediction")
print("Linear Regression + 5-Fold CV")
print("=" * 60)

X, y = make_regression(n_samples=1000, n_features=5, noise=20, random_state=42)
model = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
print(f"R2 scores for each fold: {scores.round(3)}")
print(f"Mean R2: {scores.mean().round(3)}")
print(f"Standard Deviation: {scores.std().round(3)}")
print("Model is stable across folds." if scores.std() < 0.05 else "Model varies - further investigation needed.")

# ─────────────────────────────────────────────────────────────
# SCENARIO 3: Student Exam Performance Prediction
# ─────────────────────────────────────────────────────────────
# A coaching institute tracks 800 students across 5 performance
# metrics (study hours, mock test scores, attendance, etc.)
# and wants to predict their final exam marks.
# Since the dataset is medium-sized, 5-Fold CV helps us check
# if the model generalizes well or just memorizes training data.
# Linear Regression is used to keep things interpretable.
print("\n" + "=" * 60)
print("SCENARIO 3: Student Exam Performance Prediction")
print("Linear Regression + 5-Fold CV")
print("=" * 60)

X, y = make_regression(n_samples=800, n_features=5, noise=10, random_state=42)
model = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
print(f"R2 scores for each fold: {scores.round(3)}")
print(f"Mean R2: {scores.mean().round(3)}")
print(f"Standard Deviation: {scores.std().round(3)}")
print("Model is stable across folds." if scores.std() < 0.05 else "Model varies - further analysis required.")

# ─────────────────────────────────────────────────────────────
# SCENARIO 4: Bias vs Variance - Student Exam Scores
# ─────────────────────────────────────────────────────────────
# We have exam scores of 30 students plotted against study hours.
# The relationship is non-linear (like a sine curve with noise).
# We try three models:
#   - Linear (degree 1)  -> too simple, high bias, underfits
#   - Polynomial deg 3   -> balanced, fits the pattern well
#   - Polynomial deg 10  -> too complex, high variance, overfits
# This shows why model complexity must be chosen carefully.
print("\n" + "=" * 60)
print("SCENARIO 4: Bias vs Variance - Student Exam Scores")
print("Linear (High Bias) vs Poly deg=3 (Balanced) vs Poly deg=10 (High Variance)")
print("=" * 60)

np.random.seed(0)
X_bv = np.linspace(0, 6, 30).reshape(-1, 1)
y_bv = (10 * np.sin(X_bv)).ravel() + np.random.normal(scale=3, size=30)

linear_model   = make_pipeline(PolynomialFeatures(1),  LinearRegression())
balanced_model = make_pipeline(PolynomialFeatures(3),  LinearRegression())
poly_model     = make_pipeline(PolynomialFeatures(10), LinearRegression())

linear_model.fit(X_bv, y_bv)
balanced_model.fit(X_bv, y_bv)
poly_model.fit(X_bv, y_bv)

X_test_bv = np.linspace(0, 6, 100).reshape(-1, 1)
print("Linear Model (High Bias)    - Train R2:", round(linear_model.score(X_bv, y_bv), 4))
print("Balanced Model (Poly deg=3) - Train R2:", round(balanced_model.score(X_bv, y_bv), 4))
print("Poly Model (High Variance)  - Train R2:", round(poly_model.score(X_bv, y_bv), 4))
print("Conclusion: Balanced model (degree 3) generalizes best to new students.")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, mdl, title, color in zip(
    axes,
    [linear_model, balanced_model, poly_model],
    ["High Bias (Underfitting)", "Balanced Model", "High Variance (Overfitting)"],
    ["red", "green", "blue"]
):
    ax.scatter(X_bv, y_bv, color="gray", label="Data")
    ax.plot(X_test_bv, mdl.predict(X_test_bv), color=color, label=title)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

plt.suptitle("Bias vs Variance: Student Exam Scores", fontsize=13)
plt.tight_layout()
plt.savefig("scenario based 2 march - 7 march/day8_bias_variance.png", dpi=100)
plt.close()
print("Chart saved: day8_bias_variance.png")

# ─────────────────────────────────────────────────────────────
# SCENARIO 5: Package Delivery System (Label Encoding)
# ─────────────────────────────────────────────────────────────
# A logistics company handles thousands of packages daily.
# Each package has a type (Electronics, Clothing, Food),
# a delivery zone (North, South, East, West), and a priority
# level (High, Medium, Low).
# Since ML models only understand numbers, we use Label Encoding
# to convert these text categories into numeric values.
print("\n" + "=" * 60)
print("SCENARIO 5: Package Delivery System - Label Encoding")
print("=" * 60)

delivery_data = pd.DataFrame({
    'Package_Type': ['Electronics', 'Clothing', 'Food', 'Electronics', 'Clothing'],
    'Delivery_Zone': ['North', 'South', 'East', 'West', 'North'],
    'Priority':      ['High', 'Low', 'Medium', 'High', 'Medium']
})

le = LabelEncoder()
delivery_data['Package_encoded']  = le.fit_transform(delivery_data['Package_Type'])
delivery_data['Zone_encoded']     = le.fit_transform(delivery_data['Delivery_Zone'])
delivery_data['Priority_encoded'] = le.fit_transform(delivery_data['Priority'])
print(delivery_data.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# SCENARIO 6: Restaurant Ordering System (One-Hot Encoding)
# ─────────────────────────────────────────────────────────────
# A restaurant chain wants to analyze orders across different
# cuisine types (Italian, Chinese, Indian, Mexican).
# Since cuisine has no natural order (Italian is not "greater"
# than Chinese), Label Encoding would be misleading.
# One-Hot Encoding creates a separate binary column for each
# cuisine type, which is the correct approach here.
print("\n" + "=" * 60)
print("SCENARIO 6: Restaurant Ordering System - One-Hot Encoding")
print("=" * 60)

restaurant_data = pd.DataFrame({
    'Cuisine':    ['Italian', 'Chinese', 'Indian', 'Italian', 'Mexican'],
    'Meal_Type':  ['Lunch', 'Dinner', 'Lunch', 'Dinner', 'Lunch'],
    'Order_Size': [2, 4, 1, 3, 2]
})

ohe = OneHotEncoder(sparse_output=False)
cuisine_encoded = ohe.fit_transform(restaurant_data[['Cuisine']])
cuisine_df = pd.DataFrame(cuisine_encoded, columns=ohe.get_feature_names_out(['Cuisine']))
result = pd.concat([restaurant_data, cuisine_df], axis=1)
print(result.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# SCENARIO 7: Employee Training & Satisfaction Survey (Ordinal Encoding)
# ─────────────────────────────────────────────────────────────
# An HR team collects employee satisfaction data (Low, Medium, High)
# and training levels (Beginner, Intermediate, Advanced).
# These categories have a clear order — High > Medium > Low.
# Ordinal Encoding is used here because it preserves that order,
# unlike One-Hot Encoding which treats all categories as equal.
print("\n" + "=" * 60)
print("SCENARIO 7: Employee Training & Satisfaction Survey")
print("Ordinal Encoding")
print("=" * 60)

employee_data = pd.DataFrame({
    'Department':     ['HR', 'IT', 'Sales', 'IT', 'HR'],
    'Satisfaction':   ['Low', 'High', 'Medium', 'High', 'Low'],
    'Training_Level': ['Beginner', 'Advanced', 'Intermediate', 'Advanced', 'Beginner']
})

ordinal_enc = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
employee_data['Satisfaction_encoded'] = ordinal_enc.fit_transform(employee_data[['Satisfaction']])
le2 = LabelEncoder()
employee_data['Department_encoded'] = le2.fit_transform(employee_data['Department'])
print(employee_data.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# SCENARIO 8: Patient Health Monitoring (Label + One-Hot Encoding)
# ─────────────────────────────────────────────────────────────
# A hospital system stores patient records with blood type (A+, B-, O+, etc.),
# medical condition (Diabetes, Hypertension, Healthy), and risk level.
# Blood type has no order -> One-Hot Encoding is used.
# Medical condition has no order either -> Label Encoding for simplicity.
# This scenario shows when to mix both encoding techniques.
print("\n" + "=" * 60)
print("SCENARIO 8: Patient Health Monitoring")
print("Label + One-Hot Encoding")
print("=" * 60)

patient_data = pd.DataFrame({
    'Blood_Type': ['A+', 'B-', 'O+', 'AB+', 'A-'],
    'Condition':  ['Diabetes', 'Hypertension', 'Healthy', 'Diabetes', 'Healthy'],
    'Risk_Level': ['High', 'Medium', 'Low', 'High', 'Low']
})

le3 = LabelEncoder()
patient_data['Condition_encoded'] = le3.fit_transform(patient_data['Condition'])
ohe2 = OneHotEncoder(sparse_output=False)
blood_encoded = ohe2.fit_transform(patient_data[['Blood_Type']])
blood_df = pd.DataFrame(blood_encoded, columns=ohe2.get_feature_names_out(['Blood_Type']))
result2 = pd.concat([patient_data, blood_df], axis=1)
print(result2.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# SCENARIO 9: Online Food Delivery App (One-Hot Encoding)
# ─────────────────────────────────────────────────────────────
# A food delivery app like Zomato wants to build a recommendation
# model. The dataset has cuisine type (Pizza, Burger, Sushi, Tacos),
# delivery time, and customer rating.
# Cuisine type is nominal (no order), so One-Hot Encoding is applied
# to convert it into binary columns before feeding into the model.
print("\n" + "=" * 60)
print("SCENARIO 9: Online Food Delivery App")
print("One-Hot Encoding for Cuisine Types")
print("=" * 60)

food_data = pd.DataFrame({
    'Cuisine_Type':  ['Pizza', 'Burger', 'Sushi', 'Pizza', 'Tacos'],
    'Delivery_Time': [30, 20, 45, 35, 25],
    'Rating':        [4.5, 4.0, 4.8, 4.2, 3.9]
})

ohe3 = OneHotEncoder(sparse_output=False)
cuisine_enc = ohe3.fit_transform(food_data[['Cuisine_Type']])
cuisine_df2 = pd.DataFrame(cuisine_enc, columns=ohe3.get_feature_names_out(['Cuisine_Type']))
result3 = pd.concat([food_data, cuisine_df2], axis=1)
print(result3.to_string(index=False))

print("\n" + "=" * 60)
print("DAY 8 - ALL SCENARIOS COMPLETE")
print("=" * 60)
