import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, RepeatedStratifiedKFold, TimeSeriesSplit
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import mlflow

# Get the absolute path to the project root directory
project_root = '/workspace'
sys.path.append(project_root)

from src.python.models.pipeline import create_pipeline

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
models = {
    'Ridge': (Ridge(), {
        'model__alpha': np.logspace(-1, 1, 20),
        'select__k': [10, 20, 30, 40, 'all'],
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
cv_methods = {
    'kfold': KFold(n_splits=5, shuffle=True, random_state=3),
    'stratified': RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=3),
    'timeseries': TimeSeriesSplit(n_splits=10)
}

def train_and_evaluate_models():
    """Train and evaluate all models using different CV methods and pipeline configurations."""
    mlflow.set_experiment('Regularization')
    results = []

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
                
                # Evaluate each pipeline variation
                for scale_type, pipeline in pipeline_variations.items():
                    search = GridSearchCV(
                        pipeline, 
                        params, 
                        scoring='neg_mean_squared_error',
                        cv=cv,
                        n_jobs=-1,
                        verbose=0
                    )
                    
                    search.fit(X, y)
                    rmse_score = np.sqrt(-search.best_score_)
                    
                    results.append({
                        'model': name,
                        'cv_method': cv_name,
                        'scale': scale_type,
                        'rmse': rmse_score,
                        'best_params': search.best_params_
                    })
                    
                    mlflow.log_metric(f"rmse_{scale_type}", rmse_score)
                    for param_name, param_value in search.best_params_.items():
                        mlflow.log_param(f"{scale_type}_{param_name}", param_value)
                    
                    print(f"Best for {name} ({scale_type}) with {cv_name} CV: RMSE={rmse_score:.4f}")

    return pd.DataFrame(results)

if __name__ == "__main__":
    results_df = train_and_evaluate_models()
    print("\nAll Results:")
    print(results_df.sort_values(['cv_method', 'rmse'])) 