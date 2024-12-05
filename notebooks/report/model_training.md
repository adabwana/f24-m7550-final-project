# Model Training

## Training Framework Overview

Our training framework implements a systematic approach to model development through **_parallel pipelines_** for duration and occupancy predictions. The architecture emphasizes **_modular design_** and **_reproducible experimentation_**, enabling independent optimization of model components while maintaining system cohesion. The implementation can be found in [`src/python/test_train/duration/TRAIN_duration.py`](https://github.com/adabwana/f24-m7550-final-project/blob/master/src/python/test_train/duration/TRAIN_duration.py) and [`src/python/test_train/occupancy/TRAIN_occupancy.py`](https://github.com/adabwana/f24-m7550-final-project/blob/master/src/python/test_train/occupancy/TRAIN_occupancy.py).

## Model Definition Architecture

The framework's foundation lies in specialized model definitions for each prediction task:

### Duration Model Definitions

```python
def get_model_definitions():
    """Define duration-specific model configurations."""
    return {
        'PenalizedSplines': (
            SplineRegressor(),
            {
                'model__n_knots': [10, 15, 20],
                'model__ridge_alpha': [0.1, 1.0, 10.0],
                'select_features__k': [10, 15, 'all']
            }
        ),
        'GradientBoosting': (
            GradientBoostingRegressor(),
            {
                'model__n_estimators': [100, 200],
                'model__learning_rate': [0.01, 0.1],
                'model__max_depth': [3, 5]
            }
        )
    }
```

The duration models prioritize **_temporal pattern recognition_** through specialized regressors and carefully tuned hyperparameter ranges.

### Occupancy Model Definitions

```python
def get_model_definitions():
    """Define occupancy-specific model configurations."""
    return {
        'PoissonRegressor': (
            PoissonRegressor(),
            {
                'model__alpha': [0.1, 1.0, 10.0],
                'select_features__k': [10, 15, 'all']
            }
        ),
        'RoundedRegressor': (
            RoundedRegressorWrapper(Ridge()),
            {
                'model__alpha': [0.1, 1.0, 10.0],
                'model__fit_intercept': [True, False]
            }
        )
    }
```

Occupancy models implement **_count-based constraints_** and **_integer prediction requirements_** through specialized wrappers and count-oriented algorithms.

## Pipeline Construction

The training pipeline implements three distinct preprocessing strategies:

```python
def get_pipeline_definitions():
    """Define preprocessing pipeline variants."""
    return {
        'vanilla': lambda model: Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ]),
        'interact_select': lambda model: Pipeline([
            ('scaler', StandardScaler()),
            ('interact', PolynomialFeatures(degree=2)),
            ('select_features', SelectKBest()),
            ('model', model)
        ]),
        'pca_lda': lambda model: Pipeline([
            ('scaler', StandardScaler()),
            ('reduce_dim', PCA(n_components=0.95)),
            ('model', model)
        ])
    }
```

Each pipeline variant addresses specific modeling challenges:
- **_Vanilla Pipeline_**: Baseline feature standardization
- **_Interaction Pipeline_**: Feature interaction discovery
- **_Dimension Reduction Pipeline_**: Multicollinearity handling

## Cross-Validation Strategy

The framework implements task-specific validation approaches:

```python
def get_cv_methods(n_samples):
    """Configure cross-validation strategies."""
    return {
        'kfold': KFold(n_splits=5, shuffle=True),
        'rolling': RollingForecastCV(
            min_train_size=int(0.6 * n_samples),
            forecast_horizon=int(0.1 * n_samples)
        ),
        'expanding': ExpandingForecastCV(
            min_train_size=int(0.6 * n_samples),
            forecast_horizon=int(0.1 * n_samples)
        )
    }
```

These strategies enable:
- **_K-Fold_**: Robust general performance estimation
- **_Rolling Window_**: Temporal dependency handling
- **_Expanding Window_**: Progressive data utilization

## Training Process Implementation

The training process orchestrates model fitting and evaluation:

```python
def train_single_model(name, scale_type, cv_name, pipeline, params, cv, X_train, y_train):
    """Execute single model training iteration."""
    search = GridSearchCV(
        pipeline, 
        params,
        scoring=rmse_scorer,
        cv=cv,
        n_jobs=-1,
        error_score='raise'
    )
    
    with mlflow.start_run(run_name=model_name):
        search.fit(X_train, y_train)
        mlflow.log_metrics({
            "rmse": -search.best_score_,
            "rmse_std": search.cv_results_['std_test_score'][search.best_index_]
        })
```

The implementation emphasizes:
- **_Parallel Processing_**: Efficient resource utilization
- **_Error Handling_**: Robust failure recovery
- **_Metric Logging_**: Comprehensive performance tracking

## MLflow Experiment Management

The framework implements systematic experiment tracking through MLflow:

```python
def setup_mlflow_experiments(experiment_base, model_names):
    """Configure MLflow experiment hierarchy."""
    for model_name in model_names:
        experiment_name = f"{experiment_base}/{model_name}"
        try:
            mlflow.create_experiment(experiment_name)
        except Exception as e:
            print(f"Experiment {experiment_name} already exists")
```

This structure enables:
- **_Hierarchical Organization_**: Nested experiment tracking
- **_Version Control_**: Systematic model iteration
- **_Parameter Tracking_**: Comprehensive configuration logging

## Task-Specific Training Optimizations

### Duration Model Training

Duration prediction implements specialized temporal modeling strategies:

```python
def train_models(X_train, y_train, X_test, y_test):
    """Main training function for duration prediction."""
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    experiment_base = "Duration_Pred"
    
    # Get duration-specific definitions
    models = get_model_definitions()  # From algorithms_duration.py
    pipelines = get_pipeline_definitions()
    cv_methods = get_cv_methods(len(X_train))
```

Key optimizations include:
- **_Temporal Feature Handling_**: Specialized preprocessing for time-based features
- **_Spline-Based Modeling_**: Non-linear temporal pattern capture
- **_Duration-Specific Metrics_**: RMSE optimization for time intervals

### Occupancy Model Training

Occupancy prediction employs count-based modeling approaches:

```python
def train_models(X_train, y_train, X_test, y_test):
    """Main training function for occupancy prediction."""
    experiment_base = "Occupancy_Pred"
    
    # Get occupancy-specific definitions
    models = get_model_definitions()  # From algorithms_occupancy.py
    pipelines = get_pipeline_definitions()
    cv_methods = get_cv_methods(len(X_train))
```

Specialized components include:
- **_Integer Constraints_**: Count-based prediction enforcement
- **_Capacity Limits_**: Maximum occupancy validation
- **_Poisson Modeling_**: Count distribution optimization

## Resource Management

The framework implements efficient resource utilization:

```python
def train_single_model(name, scale_type, cv_name, pipeline, params, cv, X_train, y_train):
    """Execute single model training iteration."""
    with joblib.parallel_backend('loky'):
        search = GridSearchCV(
            pipeline, 
            params,
            scoring=rmse_scorer,
            cv=cv,
            n_jobs=-1,
            error_score='raise'
        )
        search.fit(X_train, y_train)
```

Key features include:
- **_Parallel Processing_**: Multi-core utilization through joblib
- **_Memory Management_**: Efficient garbage collection
- **_Error Recovery_**: Robust failure handling

## Model Persistence Strategy

The framework implements systematic model versioning and storage:

```python
def evaluate_final_models(results, X_test, y_test):
    """Evaluate and persist final model versions."""
    final_results = []
    
    for result in results:
        model_name = f"{result['model']}_{result['pipeline_type']}_{result['cv_method']}"
        try:
            model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")
            y_pred = model.predict(X_test)
            
            final_results.append({
                'Model': result['model'],
                'Pipeline': result['pipeline_type'],
                'CV_Method': result['cv_method'],
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'R2': r2_score(y_test, y_pred)
            })
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}")
            continue
```

The persistence strategy ensures:
- **_Version Control_**: Systematic model iteration tracking
- **_Reproducibility_**: Complete parameter logging
- **_Deployment Readiness_**: Production-ready model storage

## Training Framework Benefits

The implementation delivers several key advantages:

1. **_Experimental Rigor_**
- Systematic hyperparameter optimization
- Comprehensive cross-validation
- Robust error handling

2. **_Reproducibility_**
- Complete parameter logging
- Systematic version control
- Comprehensive experiment tracking

3. **_Scalability_**
- Efficient resource utilization
- Parallel processing capabilities
- Modular component design

This training framework provides a robust foundation for model development, enabling systematic optimization while maintaining reproducibility and scalability.