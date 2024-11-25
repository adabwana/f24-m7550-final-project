import sys
import os

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, TimeSeriesSplit, RepeatedStratifiedKFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler, SplineTransformer, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import make_scorer

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

# Time series train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False  # Maintains chronological order
)

# =============================================================================
# MLFLOW CONFIGURATION
# =============================================================================
# Set up MLflow tracking URI and artifact locations
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_tracking_uri(f"sqlite:///{project_root}/mlflow.db")
os.environ["MLFLOW_TRACKING_DIR"] = f"{project_root}/mlruns"

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================
models = {
    'Ridge': (Ridge(), {
        'model__alpha': np.logspace(0, 2, 10),
    }),
    'Lasso': (Lasso(), {
        'model__alpha': np.logspace(-2, 0, 10)
    }),
    'ElasticNet': (ElasticNet(), {
        'model__alpha': np.logspace(-3, -1, 10),
        'model__l1_ratio': np.linspace(0.1, 0.9, 5)
    }),
    'PenalizedSplines': (Pipeline([
        ('spline', SplineTransformer()),
        ('ridge', Ridge())
    ]), {
        'model__spline__n_knots': [9, 11, 13, 15],
        'model__spline__degree': [3],
        'model__ridge__alpha': np.logspace(0, 2, 20)
    }),
    'KNN': (KNeighborsRegressor(), {
        'model__n_neighbors': np.arange(15, 22, 2), # Creates [15, 17, 19, 21]
        'model__weights': ['uniform', 'distance'],
        # 'model__metric': ['euclidean', 'manhattan']
    })
}

# =============================================================================
# CROSS VALIDATION SETUP
# =============================================================================
# Define stratified k-fold cross-validation
# kfold_stratified_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=3)

# Define standard k-fold cross-validation
kfold_cv = KFold(
    n_splits=10, 
    shuffle=True, 
    random_state=3
)

# Calculate default sizes based on data
n_samples = len(X_train)
n_splits = 10                                     # Too many splits=10 for number of samples=9387 with test_size=1066 and gap=0.
default_test_size = n_samples // (n_splits + 1)  # Using n_splits=10

# Define rolling window cross-validation
rolling_cv = TimeSeriesSplit(
    n_splits=n_splits,
    max_train_size=default_test_size * 5,  # Fixed training window size (rolling)
    test_size=default_test_size            # Fixed test window size
)

# Define expanding window cross-validation
expanding_cv = TimeSeriesSplit(
    n_splits=n_splits,
    test_size=default_test_size         # Fixed test window size
    # No max_train_size means the training window will expand
)

# cv_methods dictionary
cv_methods = {
    'kfold': kfold_cv,
    'rolling': rolling_cv,
    'expanding': expanding_cv
}

# =============================================================================
# MLFLOW EXPERIMENT SETUP
# =============================================================================
experiment_base = "Duration_Prediction"  # Root experiment name

# Create parent experiment if it doesn't exist
if mlflow.get_experiment_by_name(experiment_base) is None:
    mlflow.create_experiment(experiment_base)

# Create child experiments for each model type
for model_name in models.keys():
    experiment_name = f"{experiment_base}/{model_name}"
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name)

mlflow.sklearn.autolog()

# =============================================================================
# MODEL TRAINING AND EVALUATION
# =============================================================================
results = []
for name, (model, params) in models.items():
    # Set the experiment for this model type
    experiment_name = f"{experiment_base}/{name}"
    mlflow.set_experiment(experiment_name)
    
    for cv_name, cv in cv_methods.items():
        with mlflow.start_run(run_name=f"{cv_name}_{name}_val"):
            mlflow.set_tag('model', name)
            mlflow.set_tag('cv_method', cv_name)
            print(f"\nTuning {name} with {cv_name} cross-validation...")
    
            pipelines = {
                # 'standardized': Pipeline([
                #     ('scale', StandardScaler()),
                #     ('model', model)
                # ]),
                'robust': Pipeline([
                    ('scale', RobustScaler()),
                    ('model', model)
                ])
            }
            
            # Define RMSE scorer
            # Need to negate the RMSE and tell sklearn that lower values are better (STRANGE BEHAVIORS)
            rmse_scorer = make_scorer(
                lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
                greater_is_better=False  # Tell sklearn that lower values are better
            )
            
            # Iterate over each pipeline
            for scale_type, pipeline in pipelines.items():
                # Grid Search
                search = GridSearchCV(
                    pipeline, 
                    params, 
                    scoring=rmse_scorer,  # Use RMSE directly vs 'neg_mean_squared_error'
                    cv=cv,                # Using the current cv method
                    n_jobs=-1,
                    verbose=0
                )
                
                # Fit the model
                search.fit(X_train, y_train)
                
                # Calculate RMSE from negative MSE
                # rmse_score = np.sqrt(-search.best_score_)    
                rmse_score = -search.best_score_
                
                # Get the standard deviation of scores for the best parameters
                best_index = search.best_index_
                cv_std = search.cv_results_['std_test_score'][best_index]

                # Store results
                results.append({
                    'model': name,
                    'cv_method': cv_name,
                    # 'scale': scale_type,
                    'rmse': rmse_score,
                    'rmse_std': cv_std,
                    'best_params': search.best_params_
                })                
                # Log metrics to MLflow
                mlflow.log_metric(f"rmse_{scale_type}", rmse_score)
                for param_name, param_value in search.best_params_.items():
                    mlflow.log_param(f"{scale_type}_{param_name}", param_value)
                
                print(f"Best for {name} ({scale_type}) with {cv_name} CV: RMSE={rmse_score:.4f}, Params={search.best_params_}")

                # Save the best model
                mlflow.sklearn.log_model(
                    search.best_estimator_,
                    "model",
                    registered_model_name=name
                )

# =============================================================================
# RESULTS ANALYSIS
# =============================================================================
results_df = pd.DataFrame(results)
print("\nAll Results:")
print(results_df.sort_values(['rmse']))