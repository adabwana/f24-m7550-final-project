import sys
import os

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RepeatedStratifiedKFold, TimeSeriesSplit
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import mlflow

# Get the absolute path to the project root directory
project_root = '/workspace'
sys.path.append(project_root)

# import prepare_data
from src.python.preprocess import prepare_data

# =============================================================================
# DATA PREPARATION
# =============================================================================
df = pd.read_csv(f'{project_root}/data/LC_engineered.csv')

target = 'Duration_In_Min'
target_2 = 'Occupancy'
features_to_drop = ['Student_IDs', 'Semester', 'Class_Standing', 'Major', 'Expected_Graduation',
                    'Course_Name', 'Course_Number', 'Course_Type', 'Course_Code_by_Thousands',
                    'Check_Out_Time', 'Session_Length_Category', target, target_2]

X, y = prepare_data(df, target, features_to_drop)

# Define column types
numeric_features = [
    'Term_Credit_Hours', 'Term_GPA', 'Total_Credit_Hours_Earned', 
    'Cumulative_GPA', 'Change_in_GPA', 'Total_Visits', 'Semester_Visits', 
    'Avg_Weekly_Visits', 'Months_Until_Graduation', 
    'Unique_Courses', 'Course_Level_Mix', 'Advanced_Course_Ratio',
    'GPA_Trend'
]

categorical_features = [
    'Degree_Type', 'Gender', 'Time_Category', 'Course_Level',
    'Course_Name_Category', 'Course_Type_Category', 'Major_Category',
    'Has_Multiple_Majors', 'GPA_Category', 'Credit_Load_Category',
    'Class_Standing_Self_Reported', 'Class_Standing_BGSU',
    'GPA_Trend_Category', 'Week_Volume'
]

datetime_features = [
    'Check_In_Date', 'Semester_Date', 'Expected_Graduation_Date'
]

boolean_features = [
    'Is_Weekend', 'Has_Multiple_Majors'
]

# =============================================================================
# MLFLOW CONFIGURATION
# =============================================================================
sqlite_uri = f"sqlite:///{project_root}/mlflow.db"
mlflow.set_tracking_uri("http://127.0.0.1:5000")
os.environ["MLFLOW_TRACKING_DIR"] = f"{project_root}/mlruns"
mlflow.sklearn.autolog()

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================
models = {
    'Ridge': (Ridge(), {
        # 'model__alpha': np.logspace(-3, 3, 7)
        # 'model__alpha': np.logspace(-2, 2, 10)
        'model__alpha': np.logspace(0, 2, 10),
    }),
    'Lasso': (Lasso(), {
        # 'model__alpha': np.logspace(-3, 3, 7)
        # 'model__alpha': np.logspace(-2, 2, 10)
        'model__alpha': np.logspace(-2, 0, 10)
    }),
    'ElasticNet': (ElasticNet(), {
        # 'model__alpha': np.logspace(-3, 3, 7),
        # 'model__l1_ratio': np.linspace(0.1, 0.9, 5)
        'model__alpha': np.logspace(-3, -1, 10),
        'model__l1_ratio': np.linspace(0.1, 0.9, 5)
    })
}

# =============================================================================
# CROSS VALIDATION SETUP
# =============================================================================
kfold_stratified_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=3)
kfold_cv = KFold(n_splits=5, shuffle=True, random_state=3)
time_series_cv = TimeSeriesSplit(n_splits=10)

# =============================================================================
# MODEL TRAINING AND EVALUATION
# =============================================================================
mlflow.set_experiment('Regularization')
results = []

# Define CV methods dictionary
cv_methods = {
    'kfold': kfold_cv,
    'stratified': kfold_stratified_cv,
    'timeseries': time_series_cv
}

for name, (model, params) in models.items():
    for cv_name, cv in cv_methods.items():
        with mlflow.start_run(run_name=f"{name}_{cv_name}"):
            mlflow.set_tag('model', name)
            mlflow.set_tag('cv_method', cv_name)
            print(f"\nTuning {name} with {cv_name} cross-validation...")
    
            pipelines = {
                'basic': Pipeline([
                    ('model', model)
                ]),
                'standardized': Pipeline([
                    ('scale', StandardScaler()),
                    ('model', model)
                ]),
                'robust': Pipeline([
                    ('scale', RobustScaler()),
                    ('model', model)
                ])
            }
            
            # Iterate over each pipeline
            for scale_type, pipeline in pipelines.items():
                # Grid Search
                search = GridSearchCV(
                    pipeline, 
                    params, 
                    scoring='neg_mean_squared_error',
                    cv=cv,  # Using the current cv method
                    n_jobs=-1,
                    verbose=0
                )
                
                # Fit the model
                search.fit(X, y)
                
                # Calculate RMSE from negative MSE
                rmse_score = np.sqrt(-search.best_score_)
                
                # Store results
                results.append({
                    'model': name,
                    'cv_method': cv_name,
                    'scale': scale_type,
                    'rmse': rmse_score,
                    'best_params': search.best_params_
                })
                
                # Log metrics to MLflow
                mlflow.log_metric(f"rmse_{scale_type}", rmse_score)
                for param_name, param_value in search.best_params_.items():
                    mlflow.log_param(f"{scale_type}_{param_name}", param_value)
                
                print(f"Best for {name} ({scale_type}) with {cv_name} CV: RMSE={rmse_score:.4f}, Params={search.best_params_}")

# =============================================================================
# RESULTS ANALYSIS
# =============================================================================
results_df = pd.DataFrame(results)
print("\nAll Results:")
print(results_df.sort_values(['cv_method', 'rmse']))