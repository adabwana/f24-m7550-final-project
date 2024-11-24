import pandas as pd
from typing import Tuple

# Global variables
target = 'Duration_In_Min'
target_2 = 'Occupancy'
features_to_drop = ['Student_IDs', 'Semester', 'Class_Standing', 'Major', 'Expected_Graduation',
                    'Course_Name', 'Course_Number', 'Course_Type', 'Course_Code_by_Thousands',
                    'Check_Out_Time', 'Session_Length_Category', target, target_2]

def get_feature_target_split(df: pd.DataFrame, target: str, features_to_drop: list) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features and target.
    
    Args:
        df: Input DataFrame containing all features and target
        target: Name of target column
        features_to_drop: List of features to exclude
        
    Returns:
        Tuple containing features DataFrame (X) and target Series (y)
    """
    X = df.drop(features_to_drop, axis=1)
    y = df[target]
    return X, y

def convert_datetime_features(X: pd.DataFrame) -> pd.DataFrame:
    """Convert date and time columns to appropriate formats and extract numerical features.
    
    Args:
        X: Input DataFrame containing datetime features
        
    Returns:
        DataFrame with processed datetime features
    """
    # Convert dates to datetime objects
    X = X.assign(
        Check_In_Date=pd.to_datetime(X['Check_In_Date'], format='%Y-%m-%d'),
        Semester_Date=pd.to_datetime(X['Semester_Date'], format='%Y-%m-%d'),
        Expected_Graduation_Date=pd.to_datetime(X['Expected_Graduation_Date'], format='%Y-%m-%d')
    )
        
    # Convert time to total minutes
    X['Check_In_Time'] = pd.to_datetime(X['Check_In_Time'], format='%H:%M:%S').dt.hour * 60 + \
                         pd.to_datetime(X['Check_In_Time'], format='%H:%M:%S').dt.minute
    
    # Drop original datetime columns
    X = X.drop(['Check_In_Date', 'Check_In_Time', 'Semester_Date', 'Expected_Graduation_Date'], axis=1)
    
    return X

def dummy(X: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical variables to dummies.
    
    Args:
        X: Input DataFrame containing categorical features
        
    Returns:
        DataFrame with dummy variables
    """
    X_dummy = pd.get_dummies(X, drop_first=True)
    return X_dummy

def prepare_data(df: pd.DataFrame, target: str, features_to_drop: list) -> Tuple[pd.DataFrame, pd.Series]:
    """Pipeline to prepare data for modeling.
    
    Args:
        df: Raw input DataFrame
        target: Name of target column
        features_to_drop: List of features to exclude
        
    Returns:
        Tuple containing processed features DataFrame (X) and target Series (y)
    """
    # Step 1: Split features and target
    X, y = get_feature_target_split(df, target, features_to_drop)
    
    # Step 2: Convert datetime features to numerical
    X = convert_datetime_features(X)
    
    # Step 3: Create dummies
    X = dummy(X)
    
    return X, y
