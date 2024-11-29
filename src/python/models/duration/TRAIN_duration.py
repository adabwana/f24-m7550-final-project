import sys
import os
import logging
import gc
from typing import Dict, Tuple, Any
import yaml
import joblib

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, TimeSeriesSplit, RepeatedStratifiedKFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler, SplineTransformer, FunctionTransformer, PolynomialFeatures, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import make_scorer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import mlflow

# Get the absolute path to the project root directory
project_root = '/workspace'
sys.path.append(project_root)

# import prepare_data
from src.python.preprocess import prepare_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# DATA PREPARATION
# =============================================================================
df = pd.read_csv(f'{project_root}/data/processed/train_engineered.csv')

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
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_tracking_uri(f"sqlite:///{project_root}/.mlflow.db")
# os.environ["MLFLOW_TRACKING_DIR"] = f"{project_root}/.mlruns"

# =============================================================================
# CROSS VALIDATION SETUP
# =============================================================================
# Define standard k-fold cross-validation
kfold_cv = KFold(
    n_splits=10, 
    shuffle=True, 
    random_state=3
)

# Calculate default sizes based on data
n_samples = len(X_train)
n_splits = 10                                     
default_test_size = n_samples // (n_splits + 1)  

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
# MODEL DEFINITIONS
# =============================================================================
models = {
    'Ridge': (Ridge(), {
        'model__alpha': np.logspace(0, 2, 10),
        'select_features__k': np.arange(70, 100, 10),
    }),
    'Lasso': (Lasso(), {
        'model__alpha': np.logspace(-2, 0, 10),
        'select_features__k': np.arange(70, 100, 10),
    }),
    'ElasticNet': (ElasticNet(), {
        'model__alpha': np.logspace(-3, -1, 10),
        'model__l1_ratio': np.linspace(0.1, 0.9, 5),
        'select_features__k': np.arange(70, 100, 10),
    }),
    'PenalizedSplines': (Pipeline([
        ('spline', SplineTransformer()),
        ('ridge', Ridge())
    ]), {
        'model__spline__n_knots': [9, 11, 13, 15],
        'model__spline__degree': [3],
        'model__ridge__alpha': np.logspace(0, 2, 20),
        'select_features__k': np.arange(70, 100, 10),
    }),
    'KNN': (KNeighborsRegressor(), {
        'model__n_neighbors': np.arange(15, 22, 2), # Creates [15, 17, 19, 21]
        'model__weights': ['uniform', 'distance'],
        # 'model__metric': ['euclidean', 'manhattan'],
        'select_features__k': np.arange(70, 100, 10),
    }),
    'PenalizedLogNormal': (Pipeline([
        # ('scaler', RobustScaler()),
        ('log_transform', FunctionTransformer(
            func=lambda x: np.log1p(np.clip(x, 1e-10, None)),  # clip to prevent log(0)
            inverse_func=lambda x: np.expm1(x)
        )),
        ('ridge', Ridge())
    ]), {
        'model__ridge__alpha': np.logspace(0, 2, 20),
        'select_features__k': np.arange(70, 100, 10),
    }),
    'PenalizedExponential': (Pipeline([
        # ('scaler', RobustScaler()),
        ('exp_transform', FunctionTransformer(
            func=lambda x: 1/np.clip(x, 1e-10, None),  # inverse link (canonical for exponential)
            inverse_func=lambda x: 1/np.clip(x, 1e-10, None)
        )),
        ('ridge', Ridge())
    ]), {
        'model__ridge__alpha': np.logspace(0, 2, 20),
        'select_features__k': np.arange(70, 100, 10),
    }),
}

# =============================================================================
# MLFLOW EXPERIMENT SETUP
# =============================================================================
experiment_base = "Duration_Pred"

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
    logger.info(f"\nStarting training for model: {name}")
    
    # Set the experiment for this model type
    experiment_name = f"{experiment_base}/{name}"
    mlflow.set_experiment(experiment_name)
    
    # Define scalers to test
    scalers = [RobustScaler(), StandardScaler(), MinMaxScaler()]
    
    pipelines = {
        'vanilla': Pipeline([
            ('scaler', 'passthrough'), 
            ('model', model)
        ]),
        'interact_select': Pipeline([
            ('scaler', 'passthrough'), 
            ('interactions', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ('select_features', SelectKBest(score_func=f_regression, k=100)),
            ('model', model)
        ]),
        'pca_lda': Pipeline([
            ('scaler', 'passthrough'), 
            ('feature_union', FeatureUnion([
                ('pca', PCA(n_components=0.95)),
                ('lda', LinearDiscriminantAnalysis(n_components=10)),
            ])),
            ('interactions', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ('select_features', SelectKBest(score_func=f_regression, k=100)),
            ('model', model)
        ])
    }

    # Add scaler parameter to the grid search
    for scale_type, pipeline in pipelines.items():
        # Create parameter grid for this pipeline
        pipeline_params = {
            'scaler': scalers,  # Add scalers to the parameter grid
            **{f'model__{key.split("model__")[1]}': value 
               for key, value in params.items() 
               if key.startswith('model__')},
            **{f'select_features__{key.split("select_features__")[1]}': value 
               for key, value in params.items() 
               if key.startswith('select_features__')}
        }

        with mlflow.start_run(run_name=f"{name}_{scale_type}") as pipeline_run:
            mlflow.set_tag('model', name)
            mlflow.set_tag('pipeline_type', scale_type)
            logger.info(f"Tuning {name} with {scale_type} pipeline...")

            # Filter valid parameters
            valid_params = {
                param_name: param_value 
                for param_name, param_value in pipeline_params.items() 
                if param_name in pipeline.get_params()
            }

            for cv_name, cv in cv_methods.items():
                with mlflow.start_run(run_name=f"{name}_{scale_type}_{cv_name}", nested=True) as cv_run:
                    mlflow.set_tag('cv_method', cv_name)
                    logger.info(f"Using {cv_name} cross-validation...")

                    rmse_scorer = make_scorer(
                        lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
                        greater_is_better=False
                    )

                    search = GridSearchCV(
                        pipeline, 
                        valid_params,
                        scoring=rmse_scorer,
                        cv=cv,
                        n_jobs=-1,
                        verbose=0,
                        error_score='raise',
                        # return_train_score=True
                    )
                    
                    try:
                        with joblib.parallel_backend('loky'):
                            search.fit(X_train, y_train)
                        
                        rmse_score = -search.best_score_
                        best_index = search.best_index_
                        cv_std = search.cv_results_['std_test_score'][best_index]

                        # Store results
                        results.append({
                            'model': name,
                            'pipeline_type': scale_type,
                            'cv_method': cv_name,
                            'rmse': rmse_score,
                            'rmse_std': cv_std,
                            'best_params': search.best_params_
                        })                

                        # Log metrics and parameters
                        mlflow.log_metric(f"rmse_{cv_name}", rmse_score)
                        mlflow.log_metric(f"rmse_std_{cv_name}", cv_std)
                        for param_name, param_value in search.best_params_.items():
                            mlflow.log_param(f"{cv_name}_{param_name}", param_value)
                        
                        logger.info(f"RMSE={rmse_score:.4f} (±{cv_std:.4f})")
                        logger.info(f"Best parameters: {search.best_params_}")

                        # Save the model
                        mlflow.sklearn.log_model(
                            search.best_estimator_,
                            "model",
                            registered_model_name=f"{name}_{scale_type}_{cv_name}"
                        )
                        
                    except Exception as e:
                        logger.error(f"Error during {name} fitting: {str(e)}")
                        mlflow.log_param("error", str(e))
                        continue
                    
                    finally:
                        # Clean up memory
                        gc.collect()

# =============================================================================
# RESULTS ANALYSIS
# =============================================================================
results_df = pd.DataFrame(results)
print("\nAll Results:")
print(results_df.sort_values(['rmse']))

def plot_model_comparison(results_df: pd.DataFrame) -> plt.Figure:
    """Create a comparison plot of model performances."""
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='model', y='rmse', hue='cv_method', data=results_df)
    plt.xticks(rotation=45)
    plt.title('Model Performance Comparison')
    plt.tight_layout()
    return plt.gcf()

def evaluate_model_on_test(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> float:
    """Evaluate a model on the test set."""
    test_pred = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, test_pred))

# Create and log comparison plot
results_df = pd.DataFrame(results)
fig = plot_model_comparison(results_df)
mlflow.log_figure(fig, "model_comparison.png")

# Perform final evaluation on test set
logger.info("\nPerforming final evaluation on test set...")
final_results = []
for result in results:
    model_name = f"{result['model']}_{result['pipeline_type']}_{result['cv_method']}"
    try:
        model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")
        test_rmse = evaluate_model_on_test(model, X_test, y_test)
        
        final_results.append({
            'model': model_name,
            'train_rmse': result['rmse'],
            'test_rmse': test_rmse,
            'rmse_difference': test_rmse - result['rmse']
        })
    except Exception as e:
        logger.error(f"Error evaluating {model_name} on test set: {str(e)}")

final_df = pd.DataFrame(final_results)
logger.info("\nFinal Test Results:")
logger.info("\n" + final_df.sort_values(['test_rmse']).to_string())

# Save final results
final_df.to_csv(f'{project_root}/src/python/models/duration/model_evaluation_results.csv', index=False)
logger.info("\nResults saved to duration/model_evaluation_results.csv")
