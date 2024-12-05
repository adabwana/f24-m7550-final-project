# Model Testing

## Testing Framework Overview

Our testing framework implements comprehensive **_model validation_** and **_performance analysis_** through systematic evaluation pipelines. While the training chapter covers model development, this framework focuses on rigorous assessment of model behavior and production readiness. The implementation can be found in [`src/python/test_train/duration/TEST_duration.py`](https://github.com/adabwana/f24-m7550-final-project/blob/master/src/python/test_train/duration/TEST_duration.py) and [`src/python/test_train/occupancy/TEST_occupancy.py`](https://github.com/adabwana/f24-m7550-final-project/blob/master/src/python/test_train/occupancy/TEST_occupancy.py).

## Test Data Management

The testing pipeline begins with careful data preparation:

```python
# Data preparation for testing
df = pd.read_csv(f'{project_root}/data/processed/train_engineered.csv')

target = 'Duration_In_Min'  # or 'Occupancy'
features_to_drop = ['Student_IDs', 'Semester', 'Class_Standing', 'Major',
                   'Expected_Graduation', 'Course_Name', 'Course_Number',
                   'Course_Type', 'Course_Code_by_Thousands',
                   'Check_Out_Time', 'Session_Length_Category', 
                   target, target_2]

X, y = prepare_data(df, target, features_to_drop)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # Maintains chronological order
)
```

This preparation ensures:
- **_Feature Consistency_**: Matching training data structure
- **_Temporal Integrity_**: Preserved time-series ordering
- **_Data Independence_**: Proper train-test separation

## Model Loading and Validation

The framework implements systematic model retrieval and validation:

```python
def check_mlflow_connection():
    """Verify MLflow server connection and configuration."""
    try:
        client = mlflow.tracking.MlflowClient()
        client.search_experiments()
        return True
    except Exception as e:
        print(f"Error connecting to MLflow server: {e}")
        return False

# Test each model variant
for experiment in experiments:
    model_name = experiment.name.split('/')[-1]
    for pipeline_type in ['vanilla', 'interact_select', 'pca_lda']:
        for cv_name in ['kfold', 'rolling', 'expanding']:
            try:
                model = mlflow.sklearn.load_model(f"models:/{full_model_name}/latest")
                print(f"    Successfully loaded model")
            except Exception as e:
                print(f"    Couldn't load model: {e}")
                continue
```

This process ensures:
- **_Model Availability_**: Systematic version checking
- **_Configuration Validation_**: Pipeline compatibility verification
- **_Error Handling_**: Robust failure recovery

## Performance Assessment

### Metric Computation

The framework implements comprehensive performance evaluation:

```python
def calculate_metrics(y_test, y_pred):
    """Compute comprehensive performance metrics."""
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }
    return metrics
```

### Results Visualization

Our testing pipeline generates systematic performance visualizations:

```python
def plot_prediction_analysis(y_test, y_pred, model_name):
    """Generate prediction analysis visualization."""
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 'r--')
    plt.title(f'Prediction Analysis: {model_name}')
    return fig

def plot_feature_importance_biplot(X_test, y_test, y_pred, feature_names, results_path):
    """Generate feature importance visualization."""
    correlations = []
    for feature in X_test.columns:
        corr = np.corrcoef(X_test[feature], y_test - y_pred)[0,1]
        correlations.append((feature, abs(corr)))
```

## Task-Specific Testing

### Duration Model Validation

Duration testing focuses on temporal accuracy:

```python
# Duration-specific testing configuration
experiment_base = "Duration_Pred"
if metrics['RMSE'] < best_model_rmse:
    best_model_rmse = metrics['RMSE']
    best_model_predictions = y_pred
    plot_feature_importance_biplot(
        X_test, y_test, y_pred, 
        X_test.columns,
        f'{project_root}/results/duration'
    )
```

Key validation aspects:
- **_Temporal Error Analysis_**: Time-based accuracy assessment
- **_Distribution Validation_**: Duration pattern verification
- **_Feature Impact Analysis_**: Time-feature importance evaluation

### Occupancy Model Validation

Occupancy testing implements count-based validation:

```python
# Occupancy-specific testing configuration
experiment_base = "Occupancy_Pred"
metrics = calculate_metrics(y_test, y_pred.round())  # Integer predictions
save_visualization_results(results_df, project_root)
```

Critical validation components:
- **_Integer Prediction_**: Count constraint verification
- **_Range Validation_**: Occupancy limit checking
- **_Peak Analysis_**: High-traffic period accuracy

## Results Management

The framework implements systematic result collection and analysis:

```python
# Process experiment results
if experiment_results:
    top_3_results = sorted(
        experiment_results, 
        key=lambda x: x['RMSE']
    )[:3]
    test_results.extend(top_3_results)

# Create results DataFrame
results_df = pd.DataFrame(test_results)
print("\nTop 3 Models per Experiment:")
print(results_df.sort_values('RMSE'))
```

This process ensures:
- **_Systematic Comparison_**: Cross-model performance analysis
- **_Result Persistence_**: Comprehensive metric logging
- **_Performance Ranking_**: Objective model selection

## Testing Framework Benefits

1. **_Comprehensive Validation_**
- Multi-metric performance assessment
- Cross-model comparison capabilities
- Systematic visualization generation

2. **_Production Readiness_**
- Robust error handling
- Performance regression detection
- Deployment validation checks

3. **_Result Interpretability_**
- Clear performance visualization
- Feature importance analysis
- Error pattern identification

This testing framework ensures thorough model validation while maintaining clear performance insights.