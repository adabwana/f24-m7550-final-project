import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_and_prepare_data():
    # Load the engineered data
    df = pd.read_csv('data/LC_engineered.csv')
    
    # # Drop any NaN values
    # df = df.dropna()
    
    # Assuming 'duration_in_minutes' is our target variable
    # Remove any non-feature columns
    target = 'Duration_In_Min'
    features_to_drop = ['Check_Out_Time', target]
    
    X = df.drop(features_to_drop, axis=1)
    y = df[target]
    
    # Convert any categorical variables to numeric using one-hot encoding
    X = pd.get_dummies(X, drop_first=True)
    
    return X, y

def perform_statistical_tests(y_true, y_pred1, y_pred2):
    """Perform statistical tests to compare models"""
    # Shapiro-Wilk test for normality
    _, p_value_sw = stats.shapiro(y_true - y_pred1)
    print(f"Shapiro-Wilk test p-value (RF): {p_value_sw}")
    
    # Paired t-test between predictions
    t_stat, p_value = stats.ttest_rel(
        np.abs(y_true - y_pred1),
        np.abs(y_true - y_pred2)
    )
    print(f"Paired t-test p-value: {p_value}")
    
    # Wilcoxon signed-rank test
    _, p_value_w = stats.wilcoxon(y_true - y_pred1, y_true - y_pred2)
    print(f"Wilcoxon test p-value: {p_value_w}")

def bootstrap_validation(model, X, y, n_iterations=1000):
    """Perform bootstrap validation"""
    scores = []
    for _ in range(n_iterations):
        # Bootstrap sampling
        indices = np.random.randint(0, len(X), len(X))
        X_boot, y_boot = X.iloc[indices], y.iloc[indices]
        
        # Out-of-bag indices
        oob_indices = np.array(list(set(range(len(X))) - set(indices)))
        X_oob, y_oob = X.iloc[oob_indices], y.iloc[oob_indices]
        
        # Train and evaluate
        model.fit(X_boot, y_boot)
        y_pred = model.predict(X_oob)
        scores.append(r2_score(y_oob, y_pred))
    
    return np.mean(scores), np.std(scores)

def evaluate_model(y_true, y_pred, model_name):
    """Evaluate model performance using multiple metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} Performance Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

def main():
    # Load and prepare data
    print("Loading and preparing data...")
    X, y = load_and_prepare_data()
    
    # Print dataset info
    print(f"\nDataset shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=3
    )
    
    # Initialize continuous models
    linear_model = LinearRegression()
    ridge_model = Ridge(
        alpha=1.0,  # regularization strength
        random_state=3
    )
    
    # K-fold Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=3)
    
    # Perform k-fold CV for both models
    linear_scores = cross_val_score(
        linear_model, X_train, y_train, 
        cv=kf, scoring='r2'
    )
    ridge_scores = cross_val_score(
        ridge_model, X_train, y_train, 
        cv=kf, scoring='r2'
    )
    
    print("Linear Regression K-fold CV scores:", linear_scores.mean(), "±", linear_scores.std())
    print("Ridge Regression K-fold CV scores:", ridge_scores.mean(), "±", ridge_scores.std())
    
    # Bootstrap validation
    linear_boot_mean, linear_boot_std = bootstrap_validation(linear_model, X_train, y_train)
    ridge_boot_mean, ridge_boot_std = bootstrap_validation(ridge_model, X_train, y_train)
    
    print("Linear Regression Bootstrap scores:", linear_boot_mean, "±", linear_boot_std)
    print("Ridge Regression Bootstrap scores:", ridge_boot_mean, "±", ridge_boot_std)
    
    # Final training and prediction
    linear_model.fit(X_train, y_train)
    ridge_model.fit(X_train, y_train)
    
    linear_pred = linear_model.predict(X_test)
    ridge_pred = ridge_model.predict(X_test)
    
    # Perform statistical tests
    perform_statistical_tests(y_test, linear_pred, ridge_pred)
    
    # Feature importance analysis (using coefficients instead of feature_importances_)
    feature_importance_linear = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(linear_model.coef_)  # Use absolute coefficients as importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features (Linear Regression):")
    print(feature_importance_linear.head(10))
    
    # Update plot title
    plt.figure(figsize=(12, 6))
    plt.bar(feature_importance_linear['feature'][:10], feature_importance_linear['importance'][:10])
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 10 Feature Importance (Linear Regression)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    # Evaluate both models
    evaluate_model(y_test, linear_pred, "Linear Regression")
    evaluate_model(y_test, ridge_pred, "Ridge Regression")
    
    # Update plot titles
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, linear_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Duration')
    plt.ylabel('Predicted Duration')
    plt.title('Linear Regression: Actual vs Predicted')
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, ridge_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Duration')
    plt.ylabel('Predicted Duration')
    plt.title('Ridge Regression: Actual vs Predicted')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
