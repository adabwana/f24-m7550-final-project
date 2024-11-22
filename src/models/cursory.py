import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_and_prepare_data():
    # Load your data here
    # Assuming your data is in a DataFrame called 'df'
    # Remove checkout_time and keep duration_in_minutes as target
    X = df.drop(['duration_in_minutes', 'checkout_time'], axis=1)
    y = df['duration_in_minutes']
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

def main():
    # Load and prepare data
    X, y = load_and_prepare_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize models
    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    xgb_model = XGBRegressor(
        n_estimators=100,
        random_state=42
    )
    
    # K-fold Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Perform k-fold CV for both models
    rf_scores = cross_val_score(
        rf_model, X_train, y_train, 
        cv=kf, scoring='r2'
    )
    xgb_scores = cross_val_score(
        xgb_model, X_train, y_train, 
        cv=kf, scoring='r2'
    )
    
    print("Random Forest K-fold CV scores:", rf_scores.mean(), "±", rf_scores.std())
    print("XGBoost K-fold CV scores:", xgb_scores.mean(), "±", xgb_scores.std())
    
    # Bootstrap validation
    rf_boot_mean, rf_boot_std = bootstrap_validation(rf_model, X_train, y_train)
    xgb_boot_mean, xgb_boot_std = bootstrap_validation(xgb_model, X_train, y_train)
    
    print("Random Forest Bootstrap scores:", rf_boot_mean, "±", rf_boot_std)
    print("XGBoost Bootstrap scores:", xgb_boot_mean, "±", xgb_boot_std)
    
    # Final training and prediction
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    
    rf_pred = rf_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)
    
    # Perform statistical tests
    perform_statistical_tests(y_test, rf_pred, xgb_pred)
    
    # Plot actual vs predicted
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, rf_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Duration')
    plt.ylabel('Predicted Duration')
    plt.title('Random Forest: Actual vs Predicted')
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, xgb_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Duration')
    plt.ylabel('Predicted Duration')
    plt.title('XGBoost: Actual vs Predicted')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
