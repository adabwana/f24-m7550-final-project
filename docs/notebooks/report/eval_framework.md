# Evaluation Framework

## Framework Overview

Our evaluation framework implements a **_hierarchical analysis strategy_** designed to extract actionable insights from model performance data. Through systematic aggregation and comparison of results, the framework enables data-driven model selection and optimization decisions. The implementation can be found in [`src/python/evaluation/model_evals.py`](https://github.com/adabwana/f24-m7550-final-project/blob/master/src/python/evaluation/model_evals.py) and [`src/python/evaluation/best_model_params.py`](https://github.com/adabwana/f24-m7550-final-project/blob/master/src/python/evaluation/best_model_params.py).

## Analysis Architecture

### Performance Aggregation Pipeline

The framework's primary component implements systematic performance analysis across multiple dimensions:

```python
def load_and_analyze(filepath, dataset_name):
    """Hierarchical model performance analysis."""
    df = pd.read_csv(filepath)
    
    # Cross-validation strategy analysis
    cv_groups = df.groupby('CV_Method')[['RMSE', 'R2']].agg(['mean', 'std'])
    cv_groups = cv_groups.sort_values(('RMSE', 'mean'))
    
    # Pipeline architecture analysis
    pipeline_groups = df.groupby('Pipeline')[['RMSE', 'R2']].agg(['mean', 'std'])
    pipeline_groups = pipeline_groups.sort_values(('RMSE', 'mean'))
    
    # Model type analysis
    model_groups = df.groupby('Model')[['RMSE', 'R2']].agg(['mean', 'std'])
    model_groups = model_groups.sort_values(('RMSE', 'mean'))
```

This hierarchical approach enables:

- **_Cross-validation Impact Analysis_**: Understanding validation strategy effectiveness
- **_Pipeline Architecture Comparison_**: Assessing preprocessing impact
- **_Model Type Performance_**: Evaluating algorithm selection

### Best Model Identification

The framework implements systematic best model selection:

```python
def get_best_model_params(eval_path, experiment_base):
    """Extract optimal model configuration."""
    # Identify best performer
    df = pd.read_csv(eval_path)
    best_row = df.loc[df['RMSE'].idxmin()]
    
    # Retrieve detailed configuration
    model_name = f"{best_row['Model']}_{best_row['Pipeline']}_{best_row['CV_Method']}"
    client = MlflowClient()
    experiment = client.get_experiment_by_name(f"{experiment_base}/{best_row['Model']}")
    
    # Get run details
    best_run = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"attributes.run_name = '{model_name}'"
    )[0]
```

This process ensures:

- **_Objective Selection_**: Performance-based model identification
- **_Configuration Retrieval_**: Complete parameter extraction
- **_Reproducibility_**: Full model lineage tracking

## Task-Specific Evaluation

### Duration Model Analysis

The framework reveals critical insights for duration prediction:

```python
def save_and_print_results(results, dataset_type):
    """Format and persist evaluation results."""
    print(f"\n====== Best Model for Duration Prediction ======")
    print(f"Model: {results['model']}")
    print(f"Pipeline: {results['pipeline']}")
    print(f"CV Method: {results['cv_method']}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"R2: {results['r2']:.4f}")
```

Key findings include:

- **_Model Selection Impact_**: Performance variation across architectures
- **_Pipeline Effectiveness_**: Preprocessing strategy comparison
- **_Validation Strategy_**: Cross-validation method influence

### Occupancy Model Analysis

For occupancy prediction, the framework provides structured insights:

```python
# Best model configuration persistence
output_file = os.path.join(output_dir, f"occupancy_best_model.json")
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
```

Critical evaluation aspects:

- **_Count Prediction Accuracy_**: Integer constraint impact
- **_Pipeline Comparison_**: Feature engineering effectiveness
- **_Validation Assessment_**: Temporal splitting influence

## Results Management

The framework implements systematic result organization:

```python
def main():
    """Execute comprehensive evaluation pipeline."""
    # Process occupancy results
    occupancy_eval = "results/occupancy/test_evaluation.csv"
    if os.path.exists(occupancy_eval):
        occ_results = get_best_model_params(occupancy_eval, "Occupancy_Pred")
        save_and_print_results(occ_results, "occupancy")
    
    # Process duration results
    duration_eval = "results/duration/test_evaluation.csv"
    if os.path.exists(duration_eval):
        dur_results = get_best_model_params(duration_eval, "Duration_Pred")
        save_and_print_results(dur_results, "duration")
```

This organization ensures:

- **_Systematic Analysis_**: Consistent evaluation across tasks
- **_Result Persistence_**: Structured output storage
- **_Reproducible Insights_**: Clear analysis lineage

## Framework Benefits

The evaluation framework delivers three primary advantages:

1. **_Comprehensive Analysis_**
- Multi-dimensional performance assessment
- Cross-model comparison methodology
- Statistical significance evaluation

2. **_Insight Generation_**
- Parameter sensitivity understanding
- Architecture impact quantification
- Validation strategy assessment

3. **_Decision Support_**
- Objective model selection
- Configuration optimization
- Deployment readiness validation

This evaluation framework provides the analytical foundation for model selection and optimization decisions, complementing the training and testing processes with rigorous performance analysis.