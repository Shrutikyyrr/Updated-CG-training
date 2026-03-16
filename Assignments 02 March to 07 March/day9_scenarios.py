"""
Day 9 - Encoding, ML Pipelines, Scikit-Learn Model Selection & GridSearchCV
============================================================================
Scenarios:
  1. Retail Sales Data Encoding (Label + One-Hot)
  2. Healthcare Patient Records Encoding (Label + One-Hot)
  3. Student Exam Pass Prediction (Pipeline)
  4. Employee Attrition Prediction (Pipeline)
  5. Customer Purchase Prediction (Pipeline + GridSearchCV)
  6. Employee Attrition Prediction (Pipeline + GridSearchCV)
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

# ─────────────────────────────────────────────────────────────
# SCENARIO 1: Retail Sales Data Encoding
# ─────────────────────────────────────────────────────────────
# You are a data analyst at a retail company. The company wants
# to build a model to predict sales performance based on product
# type and region. But the dataset has text columns — Product
# (Shoes, Shirt, Watch) and Region (North, South, East, West).
# ML models can't read text, so we need to convert them to numbers.
# Label Encoding is used for Product (ordinal-like),
# One-Hot Encoding is used for Region (no natural order).
print("=" * 60)
print("SCENARIO 1: Retail Sales Data Encoding")
print("Label Encoding + One-Hot Encoding")
print("=" * 60)

data = pd.DataFrame({
    'Product': ['Shoes', 'Shirt', 'Shoes', 'Watch'],
    'Region':  ['North', 'South', 'East', 'West'],
    'Sales':   [200, 150, 300, 400]
})

le = LabelEncoder()
data['Product_encoded'] = le.fit_transform(data['Product'])

ohe = OneHotEncoder(sparse_output=False)
region_encoded = ohe.fit_transform(data[['Region']])
region_df = pd.DataFrame(region_encoded, columns=ohe.get_feature_names_out(['Region']))
data = pd.concat([data, region_df], axis=1)
print(data)

# ─────────────────────────────────────────────────────────────
# SCENARIO 2: Healthcare Patient Records Encoding
# ─────────────────────────────────────────────────────────────
# A hospital wants to predict patient recovery time based on
# treatment type and hospital wing. The dataset has:
#   - Treatment_Type: Surgery, Therapy, Medication (categorical)
#   - Hospital_Wing: East, West, North, South (categorical)
#   - Recovery_Days: target variable (numeric)
# Treatment type is label-encoded, hospital wing is one-hot encoded.
# This prepares the data for any ML regression model.
print("\n" + "=" * 60)
print("SCENARIO 2: Healthcare Patient Records Encoding")
print("Label Encoding + One-Hot Encoding")
print("=" * 60)

data = pd.DataFrame({
    'Treatment_Type': ['Surgery', 'Therapy', 'Medication', 'Surgery', 'Therapy'],
    'Hospital_Wing':  ['East', 'West', 'North', 'South', 'East'],
    'Recovery_Days':  [15, 10, 7, 20, 12]
})

le = LabelEncoder()
data['Treatment_encoded'] = le.fit_transform(data['Treatment_Type'])

ohe = OneHotEncoder(sparse_output=False)
wing_encoded = ohe.fit_transform(data[['Hospital_Wing']])
wing_df = pd.DataFrame(wing_encoded, columns=ohe.get_feature_names_out(['Hospital_Wing']))
data = pd.concat([data, wing_df], axis=1)
print("\nEncoded Data:\n")
print(data)

# ─────────────────────────────────────────────────────────────
# SCENARIO 3: Student Exam Pass Prediction (Pipeline)
# ─────────────────────────────────────────────────────────────
# A university wants to predict whether a student will pass or
# fail based on age, study hours, gender, and school type.
# The challenge: numeric and categorical features need different
# preprocessing. Instead of doing it manually in steps, we build
# a single Pipeline that:
#   - Scales numeric features (age, hours_study) with StandardScaler
#   - Encodes categorical features (gender, school) with OneHotEncoder
#   - Trains a Logistic Regression classifier
# This keeps the code clean and prevents data leakage.
print("\n" + "=" * 60)
print("SCENARIO 3: Student Exam Pass Prediction")
print("ML Pipeline (StandardScaler + OneHotEncoder + LogisticRegression)")
print("=" * 60)

data = {
    'age':         [18, 19, 18, 20, 21],
    'hours_study': [2, 5, 1, 6, 4],
    'gender':      ['Male', 'Female', 'Female', 'Male', 'Male'],
    'school':      ['Government', 'Private', 'Government', 'Private', 'Government'],
    'pass_exam':   [0, 1, 0, 1, 1]
}
df = pd.DataFrame(data)

X = df[['age', 'hours_study', 'gender', 'school']]
y = df['pass_exam']

numeric_features     = ['age', 'hours_study']
categorical_features = ['gender', 'school']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(), categorical_features)
])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier',   LogisticRegression())
])

pipeline.fit(X, y)

new_student = pd.DataFrame({
    'age': [19], 'hours_study': [3],
    'gender': ['Female'], 'school': ['Government']
})
prediction = pipeline.predict(new_student)
print("Prediction (1=Pass, 0=Fail):", prediction)

# ─────────────────────────────────────────────────────────────
# SCENARIO 4: Employee Attrition Prediction (Pipeline)
# ─────────────────────────────────────────────────────────────
# An HR team wants to predict whether an employee will leave the
# company (attrition = 1) or stay (attrition = 0).
# Features: age, years of experience (numeric),
#           department, education level (categorical).
# A Pipeline is built to handle preprocessing and classification
# in one go. This is the standard industry approach for building
# clean, reproducible ML workflows.
print("\n" + "=" * 60)
print("SCENARIO 4: Employee Attrition Prediction")
print("ML Pipeline (StandardScaler + OneHotEncoder + LogisticRegression)")
print("=" * 60)

data = {
    'age':              [25, 30, 28, 35, 40],
    'years_experience': [2, 5, 4, 10, 12],
    'department':       ['HR', 'IT', 'Sales', 'IT', 'HR'],
    'education':        ['Graduate', 'Postgraduate', 'Graduate', 'Postgraduate', 'Graduate'],
    'attrition':        [1, 0, 1, 0, 0]
}
df = pd.DataFrame(data)

X = df[['age', 'years_experience', 'department', 'education']]
y = df['attrition']

numeric_features     = ['age', 'years_experience']
categorical_features = ['department', 'education']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(), categorical_features)
])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier',   LogisticRegression())
])

pipeline.fit(X, y)

new_employee = pd.DataFrame({
    'age': [32], 'years_experience': [6],
    'department': ['Sales'], 'education': ['Graduate']
})
prediction = pipeline.predict(new_employee)
print("Prediction (1=Leaves, 0=Stays):", prediction)

# ─────────────────────────────────────────────────────────────
# SCENARIO 5: Customer Purchase Prediction (Pipeline + GridSearchCV)
# ─────────────────────────────────────────────────────────────
# A retail company wants to predict if a customer will buy a product
# based on age, income, gender, and city.
# The team already has a Pipeline ready. Now they want to find the
# best hyperparameters for Logistic Regression (C value, solver).
# GridSearchCV tries all combinations and picks the best one using
# 3-fold cross-validation. This automates the tuning process.
print("\n" + "=" * 60)
print("SCENARIO 5: Customer Purchase Prediction")
print("Pipeline + GridSearchCV (Hyperparameter Tuning)")
print("=" * 60)

data = {
    'age':       [25, 30, 35, 40, 22, 28, 45, 50, 32, 38],
    'income':    [30000, 50000, 70000, 60000, 35000, 48000, 90000, 80000, 55000, 65000],
    'gender':    ['Male', 'Female', 'Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male'],
    'city':      ['New York', 'Los Angeles', 'Chicago', 'Houston', 'New York',
                  'Chicago', 'Los Angeles', 'Houston', 'New York', 'Chicago'],
    'purchased': [0, 1, 0, 1, 0, 1, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

X = df.drop("purchased", axis=1)
y = df["purchased"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_features     = ["age", "income"]
categorical_features = ["gender", "city"]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier",   LogisticRegression(max_iter=1000))
])

param_grid = {
    "classifier__C":      [0.01, 0.1, 1, 10],
    "classifier__solver": ["lbfgs", "saga"]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))

# ─────────────────────────────────────────────────────────────
# SCENARIO 6: Employee Attrition Prediction (Pipeline + GridSearchCV)
# ─────────────────────────────────────────────────────────────
# Same HR attrition problem, but now the team wants to go further
# and tune the model using GridSearchCV with 5-fold CV.
# They test different values of C (0.1, 1, 10) and two solvers
# (liblinear, lbfgs) to find the combination that gives the
# highest accuracy. This is standard practice before deploying
# any classification model in production.
print("\n" + "=" * 60)
print("SCENARIO 6: Employee Attrition Prediction")
print("Pipeline + GridSearchCV (Hyperparameter Tuning)")
print("=" * 60)

data = {
    'age':              [25, 30, 28, 35, 40, 45, 50, 29],
    'years_experience': [2, 7, 5, 10, 15, 20, 25, 4],
    'department':       ['HR', 'IT', 'Sales', 'IT', 'HR', 'Sales', 'IT', 'HR'],
    'education_level':  ['Bachelor', 'Master', 'Bachelor', 'PhD', 'Master', 'Bachelor', 'PhD', 'Master'],
    'attrition':        [0, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

X = df.drop("attrition", axis=1)
y = df["attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_features     = ["age", "years_experience"]
categorical_features = ["department", "education_level"]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier",   LogisticRegression(max_iter=1000))
])

param_grid = {
    "classifier__C":      [0.1, 1, 10],
    "classifier__solver": ["liblinear", "lbfgs"]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
y_pred = grid_search.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

print("\n" + "=" * 60)
print("DAY 9 - ALL SCENARIOS COMPLETE")
print("=" * 60)
