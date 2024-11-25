import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA

def prepare_features(df):
    """Prepare features from the input DataFrame."""
    # Convert datetime columns
    datetime_cols = ['Check_In_Date', 'Semester_Date', 'Expected_Graduation_Date']
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    # Convert Check_In_Time to minutes since midnight
    df['Check_In_Time'] = pd.to_datetime(df['Check_In_Time']).dt.hour * 60 + pd.to_datetime(df['Check_In_Time']).dt.minute
    
    # Define targets and features to drop
    target = 'Duration_In_Min'
    target_2 = 'Occupancy'
    features_to_drop = ['Student_IDs', 'Semester', 'Class_Standing', 'Major', 'Expected_Graduation',
                       'Course_Name', 'Course_Number', 'Course_Type', 'Course_Code_by_Thousands',
                       'Check_Out_Time', 'Session_Length_Category', target, target_2]
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in features_to_drop]
    
    # Detect column types
    numeric_features = df[feature_cols].select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_features = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    boolean_features = df[feature_cols].select_dtypes(include=['bool']).columns.tolist()
    datetime_features = df[feature_cols].select_dtypes(include=['datetime64', 'timedelta64']).columns.tolist()
    
    # Combine boolean and categorical features
    categorical_features = categorical_features + boolean_features
    
    return numeric_features, categorical_features

def create_pipeline(model, numeric_features, categorical_features, include_pca=False, feature_selection=False):
    """Create a sklearn pipeline with basic preprocessing steps."""
    
    # Basic preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
    )
    
    # Build pipeline steps
    steps = [('preprocessor', preprocessor)]
    
    if include_pca:
        steps.append(('pca', PCA(n_components=0.95)))
    
    if feature_selection:
        steps.append(('select', SelectKBest(score_func=f_regression, k=20)))
    
    steps.append(('model', model))
    
    return Pipeline(steps)

