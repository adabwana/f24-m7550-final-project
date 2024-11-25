import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns

# Get the absolute path to the project root directory
project_root = '/workspace'
sys.path.append(project_root)

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

# Add error handling for MLflow connection
def check_mlflow_connection():
    try:
        client = mlflow.tracking.MlflowClient()
        client.search_experiments()
        return True
    except Exception as e:
        print(f"Error connecting to MLflow server: {e}")
        print("Please ensure MLflow server is running with:")
        print("mlflow server --backend-store-uri sqlite:///workspace/mlflow.db " + 
              "--default-artifact-root file:///workspace/mlruns --host 0.0.0.0 --port 5000")
        return False

# Check MLflow connection before proceeding
if not check_mlflow_connection():
    sys.exit(1)

def calculate_metrics(y_true, y_pred):
    """Calculate multiple regression metrics."""
    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }
    return metrics

def plot_prediction_analysis(y_true, y_pred, model_name):
    """Create detailed prediction analysis plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Prediction Analysis for {model_name}', fontsize=16)
    
    # Scatter plot of predicted vs actual values
    axes[0, 0].scatter(y_true, y_pred, alpha=0.5)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('Predicted vs Actual Values')
    
    # Residuals plot
    residuals = y_pred - y_true
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals vs Predicted Values')
    
    # Histogram of residuals
    axes[1, 0].hist(residuals, bins=30)
    axes[1, 0].set_xlabel('Residual Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Residuals')
    
    # Q-Q plot of residuals
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot of Residuals')
    
    plt.tight_layout()
    return fig

# =============================================================================
# MODEL TESTING
# =============================================================================
experiment_base = "Duration_Prediction"
client = mlflow.tracking.MlflowClient()

# Get all experiments
experiments = client.search_experiments(
    filter_string=f"name LIKE '{experiment_base}/%'"
)

test_results = []
best_model_info = {'rmse': float('inf')}

# Create results directory if it doesn't exist
os.makedirs(f'{project_root}/results', exist_ok=True)

# Test each model and find the best one
for experiment in experiments:
    model_name = experiment.name.split('/')[-1]
    
    # Try each CV method
    cv_methods = ['kfold', 'rolling', 'expanding']
    
    for cv_name in cv_methods:
        full_model_name = f"{cv_name}_{model_name}"
        print(f"\nTesting {full_model_name}...")
        
        try:
            # Try to load the registered model with CV prefix
            model = mlflow.sklearn.load_model(f"models:/{full_model_name}/latest")
            print(f"Successfully loaded registered model {full_model_name}")
            
            # Make predictions and evaluate
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                metrics = calculate_metrics(y_test, y_pred)
                
                # Store results
                result = {
                    'Model': model_name,
                    'CV_Method': cv_name,
                    'MSE': metrics['MSE'],
                    'RMSE': metrics['RMSE'],
                    'MAE': metrics['MAE'],
                    'R2': metrics['R2']
                }
                test_results.append(result)
                
                print(f"Results for {cv_name}_{model_name}:")
                for metric_name, metric_value in metrics.items():
                    print(f"{metric_name}: {metric_value:.4f}")
                
                # Check if this is the best model so far
                if metrics['RMSE'] < best_model_info['rmse']:
                    best_model_info = {
                        'model_name': model_name,
                        'cv_method': cv_name,
                        'rmse': metrics['RMSE'],
                        'predictions': y_pred,
                        'model': model
                    }
                
                # Create and save prediction analysis plots
                fig = plot_prediction_analysis(y_test, y_pred, f"{cv_name}_{model_name}")
                plt.savefig(f'{project_root}/results/{cv_name}_{model_name}_prediction_analysis.png')
                plt.close()
                
            except Exception as e:
                print(f"Error during prediction and evaluation: {e}")
                continue
                
        except Exception as e1:
            print(f"Couldn't load registered model {full_model_name}: {e1}")
            continue

if not test_results:
    print("\nNo models were successfully tested!")
    sys.exit(1)

# Create results DataFrame
results_df = pd.DataFrame(test_results)
print("\nAll Results:")
print(results_df.sort_values('RMSE'))

# Save results to CSV
results_df.to_csv(f'{project_root}/results/model_comparison.csv', index=False)

# Plot comparison of model performances
plt.figure(figsize=(12, 6))
for metric in ['RMSE', 'MAE', 'R2']:
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x='Model', y=metric, hue='CV_Method')
    plt.title(f'{metric} by Model and CV Method')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{project_root}/results/{metric}_comparison.png')
    plt.close()

# Print best model details with CV method
if best_model_info['model_name']:
    print(f"\nBest Model: {best_model_info['cv_method']}_{best_model_info['model_name']}")
    print(f"RMSE: {best_model_info['rmse']:.4f}")

