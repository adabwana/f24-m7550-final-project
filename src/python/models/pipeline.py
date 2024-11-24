import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, PowerTransformer,
    OneHotEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA

# Custom transformer for datetime features
class DateTimeFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Convert Check_In_Time to minutes since midnight
        if 'Check_In_Time' in X.columns:
            X['Check_In_Time'] = pd.to_datetime(X['Check_In_Time']).dt.hour * 60 + \
                                pd.to_datetime(X['Check_In_Time']).dt.minute
        
        # Convert dates to months until graduation
        if 'Expected_Graduation_Date' in X.columns and 'Check_In_Date' in X.columns:
            X['Expected_Graduation_Date'] = pd.to_datetime(X['Expected_Graduation_Date'])
            X['Check_In_Date'] = pd.to_datetime(X['Check_In_Date'])
            X['Months_To_Graduation'] = (X['Expected_Graduation_Date'] - X['Check_In_Date']).dt.days / 30
            
            # Drop original date columns after extraction
            X = X.drop(['Expected_Graduation_Date', 'Check_In_Date'], axis=1)
        
        return X

# Define column types
numeric_features = [
    'Term_Credit_Hours', 'Term_GPA', 'Total_Credit_Hours_Earned', 
    'Cumulative_GPA', 'Change_in_GPA', 'Total_Visits', 'Semester_Visits', 
    'Avg_Weekly_Visits', 'Week_Volume', 'Months_Until_Graduation', 
    'Unique_Courses', 'Course_Level_Mix', 'Advanced_Course_Ratio'
]

categorical_features = [
    'Degree_Type', 'Gender', 'Time_Category', 'Course_Level',
    'Course_Name_Category', 'Course_Type_Category', 'Major_Category',
    'Has_Multiple_Majors', 'GPA_Category', 'Credit_Load_Category',
    'Class_Standing_Self_Reported', 'Class_Standing_BGSU',
    'GPA_Trend_Category'
]

datetime_features = ['Check_In_Date', 'Semester_Date', 'Expected_Graduation_Date']
boolean_features = ['Is_Weekend', 'Has_Multiple_Majors']

def create_pipeline(model, include_pca=True, feature_selection=True):
    """Create a sklearn pipeline with preprocessing, optional PCA and feature selection.
    
    Args:
        model: The sklearn model to use
        include_pca: Whether to include PCA dimensionality reduction
        feature_selection: Whether to include feature selection
        
    Returns:
        A sklearn Pipeline object
    """
    # Preprocessing for numerical features
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('power', PowerTransformer(standardize=True))
    ])

    # Preprocessing for categorical features
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    # Boolean features transformer
    boolean_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=False))
    ])

    # Combine all preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('bool', boolean_transformer, boolean_features),
            ('date', DateTimeFeatureTransformer(), datetime_features)
        ],
        remainder='passthrough'
    )

    # Create the full pipeline
    steps = [('preprocessor', preprocessor)]
    
    if include_pca:
        steps.append(('pca', PCA(n_components=0.95)))
    
    if feature_selection:
        steps.append(('select', SelectKBest(score_func=f_regression, k=20)))
    
    steps.append(('model', model))
    
    return Pipeline(steps) 