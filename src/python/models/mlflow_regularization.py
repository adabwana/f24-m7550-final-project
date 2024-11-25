import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import mlflow

# Get the absolute path to the project root directory
project_root = '/workspace'
sys.path.append(project_root)

from src.python.models.pipeline import create_pipeline, prepare_features

# =============================================================================
# DATA PREPARATION
# =============================================================================
df = pd.read_csv(f'{project_root}/data/LC_engineered.csv')

# Get numeric and categorical features
numeric_features, categorical_features = prepare_features(df)

target = 'Duration_In_Min'
target_2 = 'Occupancy'
features_to_drop = ['Student_IDs', 'Semester', 'Class_Standing', 'Major', 'Expected_Graduation',
                    'Course_Name', 'Course_Number', 'Course_Type', 'Course_Code_by_Thousands',
                    'Check_Out_Time', 'Session_Length_Category', target, target_2]

# Simple split without preprocessing (Pipeline will handle it)
X = df.drop(features_to_drop, axis=1)
y = df[target]

# =============================================================================
# MLFLOW CONFIGURATION
# =============================================================================
mlflow.set_tracking_uri("http://127.0.0.1:5000")
os.environ["MLFLOW_TRACKING_DIR"] = f"{project_root}/mlruns"
# Configure autolog with specific parameters
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    silent=False
)

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================
models = {
    'Ridge': (Ridge(), {
        'model__alpha': [1.0],
    }),
    'Lasso': (Lasso(), {
        'model__alpha': [1.0],
    }),
    'ElasticNet': (ElasticNet(), {
        'model__alpha': [1.0],
        'model__l1_ratio': [0.5],
    })
}

# =============================================================================
# CROSS VALIDATION SETUP AND TRAINING
# =============================================================================
cv = KFold(n_splits=5, shuffle=True, random_state=42)

def train_and_evaluate_models():
    """Train and evaluate all models using different CV methods and pipeline configurations."""
    mlflow.set_experiment('Regularization')
    results = []

    for name, (model, params) in models.items():
        with mlflow.start_run(run_name=f"{name}"):
            mlflow.set_tag('model', name)
            print(f"\nTuning {name}...")
    
            pipeline = create_pipeline(
                model=model,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                include_pca=False,
                feature_selection=False
            )
            
            search = GridSearchCV(
                pipeline, 
                params, 
                scoring='neg_mean_squared_error',
                cv=cv,
                n_jobs=-1,
                verbose=1
            )
            
            try:
                search.fit(X, y)
                rmse_score = np.sqrt(-search.best_score_)
                
                # Store results with all necessary columns
                results.append({
                    'model_name': name,
                    'rmse': rmse_score,
                    'best_params': search.best_params_,
                    'best_score': search.best_score_
                })
                
                # Log to MLflow
                mlflow.log_metric("rmse", rmse_score)
                for param_name, param_value in search.best_params_.items():
                    mlflow.log_param(param_name, param_value)
                
                print(f"Best for {name}: RMSE={rmse_score:.4f}")
            
            except Exception as e:
                print(f"Error with {name}: {str(e)}")
                continue

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort and display results
    if not results_df.empty:
        sorted_results = results_df.sort_values('rmse', ascending=True)
        print("\nAll Results:")
        print(sorted_results[['model_name', 'rmse', 'best_score']].to_string())
    else:
        print("\nNo results to display")
    
    return results_df

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    results_df = train_and_evaluate_models() 