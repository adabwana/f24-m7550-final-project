import sys
import os

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RepeatedStratifiedKFold, TimeSeriesSplit
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, PowerTransformer,
    OneHotEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import mlflow

# Get the absolute path to the project root directory
project_root = '/workspace'
sys.path.append(project_root)

# -----------------------------------------------------------------------------
# DATA PREPARATION
# -----------------------------------------------------------------------------
df = pd.read_csv(f'{project_root}/data/LC_engineered.csv')

target = 'Duration_In_Min'
target_2 = 'Occupancy'
features_to_drop = ['Student_IDs', 'Semester', 'Class_Standing', 'Major', 'Expected_Graduation',
                    'Course_Name', 'Course_Number', 'Course_Type', 'Course_Code_by_Thousands',
                    'Check_Out_Time', 'Session_Length_Category', target, target_2]

# Simple split without preprocessing (Pipeline will handle it)
X = df.drop(features_to_drop, axis=1)
y = df[target]

# -----------------------------------------------------------------------------
# MLFLOW CONFIGURATION
# -----------------------------------------------------------------------------
sqlite_uri = f"sqlite:///{project_root}/mlflow.db"
mlflow.set_tracking_uri("http://127.0.0.1:5000")
os.environ["MLFLOW_TRACKING_DIR"] = f"{project_root}/mlruns"
mlflow.sklearn.autolog()

# -----------------------------------------------------------------------------
# MODEL DEFINITIONS
# -----------------------------------------------------------------------------
# Custom transformer for datetime features
class DateTimeFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Convert Check_In_Time to minutes since midnight
        if 'Check_In_Time' in X.columns:
            X['Check_In_Time'] = pd.to_datetime(X['Check_In_Time']).dt.hour * 60 + \
                                pd.to_datetime(X['Check_In_Time']).dt.minute
        
        # Convert dates to months until graduation
        if 'Expected_Graduation_Date' in X.columns and 'Check_In_Date' in X.columns:
            X['Expected_Graduation_Date'] = pd.to_datetime(X['Expected_Graduation_Date'])
            X['Check_In_Date'] = pd.to_datetime(X['Check_In_Date'])
            X['Months_To_Graduation'] = (X['Expected_Graduation_Date'] - X['Check_In_Date']).dt.days / 30
            
            # Drop original date columns after extraction
            X = X.drop(['Expected_Graduation_Date', 'Check_In_Date'], axis=1)
        
        return X

# Define column types based on your data
numeric_features = [
    'Term_Credit_Hours', 'Term_GPA', 'Total_Credit_Hours_Earned', 
    'Cumulative_GPA', 'Change_in_GPA', 'Total_Visits', 'Semester_Visits', 
    'Avg_Weekly_Visits', 'Week_Volume', 'Months_Until_Graduation', 
    'Unique_Courses', 'Course_Level_Mix', 'Advanced_Course_Ratio'
]

categorical_features = [
    'Degree_Type', 'Gender', 'Time_Category', 'Course_Level',
    'Course_Name_Category', 'Course_Type_Category', 'Major_Category',
    'Has_Multiple_Majors', 'GPA_Category', 'Credit_Load_Category',
    'Class_Standing_Self_Reported', 'Class_Standing_BGSU',
    'GPA_Trend_Category'
]

datetime_features = ['Check_In_Date', 'Semester_Date', 'Expected_Graduation_Date']

# Boolean features (can be treated as categorical or numeric)
boolean_features = ['Is_Weekend', 'Has_Multiple_Majors']

# Create the pipeline
def create_pipeline(model, include_pca=True, feature_selection=True):
    # Preprocessing for numerical features
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('power', PowerTransformer(standardize=True))
    ])

    # Preprocessing for categorical features
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    # Boolean features can be left as is or minimally processed
    boolean_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=False))
    ])

    # Combine all preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('bool', boolean_transformer, boolean_features),
            ('date', DateTimeFeatureTransformer(), datetime_features)
        ],
        remainder='passthrough'
    )

    # Create the full pipeline
    steps = [('preprocessor', preprocessor)]
    
    if include_pca:
        steps.append(('pca', PCA(n_components=0.95)))
    
    if feature_selection:
        steps.append(('select', SelectKBest(score_func=f_regression, k=20)))
    
    steps.append(('model', model))
    
    return Pipeline(steps)

models = {
    'Ridge': (Ridge(), {
        'model__alpha': np.logspace(-1, 1, 20),
        'select__k': [10, 20, 30, 40, 'all'],  # Feature selection options
        'preprocessor__num__power__method': ['yeo-johnson', 'box-cox'],
        'pca__n_components': [0.90, 0.95, 0.99]
    }),
    'Lasso': (Lasso(), {
        'model__alpha': np.logspace(-1, 1, 20),
        'select__k': [10, 20, 30, 40, 'all'],
        'preprocessor__num__power__method': ['yeo-johnson', 'box-cox'],
        'pca__n_components': [0.90, 0.95, 0.99]
    }),
    'ElasticNet': (ElasticNet(), {
        'model__alpha': np.logspace(-1, 1, 20),
        'model__l1_ratio': np.linspace(0.1, 0.9, 5),
        'select__k': [10, 20, 30, 40, 'all'],
        'preprocessor__num__power__method': ['yeo-johnson', 'box-cox'],
        'pca__n_components': [0.90, 0.95, 0.99]
    })
}

# -----------------------------------------------------------------------------
# CROSS VALIDATION SETUP
# -----------------------------------------------------------------------------
kfold_stratified_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=3)
kfold_cv = KFold(n_splits=5, shuffle=True, random_state=3)
time_series_cv = TimeSeriesSplit(n_splits=10)

# -----------------------------------------------------------------------------
# MODEL TRAINING AND EVALUATION
# -----------------------------------------------------------------------------
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
    
            # Create pipeline variations
            pipeline_variations = {
                'basic': create_pipeline(model, include_pca=False, feature_selection=False),
                'with_pca': create_pipeline(model, include_pca=True, feature_selection=False),
                'with_selection': create_pipeline(model, include_pca=False, feature_selection=True),
                'full': create_pipeline(model, include_pca=True, feature_selection=True)
            }
            
            # Iterate over each pipeline
            for scale_type, pipeline in pipeline_variations.items():
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

# -----------------------------------------------------------------------------
# RESULTS ANALYSIS
# -----------------------------------------------------------------------------
results_df = pd.DataFrame(results)
print("\nAll Results:")
print(results_df.sort_values(['cv_method', 'rmse']))
