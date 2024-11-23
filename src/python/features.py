# Load required libraries
import pandas as pd
from pathlib import Path
import numpy as np

# -----------------------------------------------------------------------------
# READ DATA
# -----------------------------------------------------------------------------
# Using Path for cross-platform compatibility
data_path = Path("data/LC_train.csv")
data_raw = pd.read_csv(data_path)

# -----------------------------------------------------------------------------
# ENGINEER FEATURES
# -----------------------------------------------------------------------------
def add_temporal_features(df):
    """Add time-based features to the dataframe."""
    return df.assign(
        Day_of_Week=df['Check_In_Date'].dt.day_name().str[:3],
        Is_Weekend=df['Check_In_Date'].dt.dayofweek.isin([5, 6]),
        Week_of_Month=df['Check_In_Date'].dt.day.apply(lambda x: np.ceil(x/7)),
        Month=df['Check_In_Date'].dt.month_name().str[:3],
        Hour_of_Day=pd.to_datetime(df['Check_In_Time'].astype(str), format='%H:%M:%S').dt.hour,
    )

def add_time_period(df):
    """Categorize hours into time periods."""
    return df.assign(
        Time_Period=pd.cut(
            df['Hour_of_Day'],
            bins=[-np.inf, 6, 12, 17, 22, np.inf],
            labels=['Late Night', 'Morning', 'Afternoon', 'Evening', 'Late Night'],
            ordered=False
        )
    )

def add_course_features(df):
    """Add course-related features."""
    df = df.copy()
    df['Course_Level'] = df['Course_Code_by_Thousands'].map({
        '1000': 'Introductory', 
        '2000': 'Intermediate',
    }).fillna('Advanced')
    df.loc[~df['Course_Level'].isin(['Introductory', 'Intermediate']), 'Course_Level'] = 'Other'
    return df

def add_performance_indicators(df):
    """Add student performance-related features."""
    return df.assign(
        GPA_Category=pd.cut(
            df['Cumulative_GPA'],
            bins=[-np.inf, 2.0, 3.0, 3.5, np.inf],
            labels=['Needs Improvement', 'Satisfactory', 'Good', 'Excellent']
        )
    )

def add_session_features(df):
    """Add study session-related features."""
    duration = ((pd.to_datetime(df['Check_Out_Time'].astype(str), format='%H:%M:%S') - 
                pd.to_datetime(df['Check_In_Time'].astype(str), format='%H:%M:%S'))
               .dt.total_seconds() / 60)
    
    return df.assign(
        Duration_In_Min=duration,
        Session_Length_Category=pd.cut(
            duration,
            bins=[-np.inf, 30, 90, 180, np.inf],
            labels=['Short', 'Medium', 'Long', 'Extended']
        )
    )

def add_credit_load_features(df):
    """Add credit load-related features."""
    return df.assign(
        Credit_Load_Category=pd.cut(
            df['Term_Credit_Hours'],
            bins=[-np.inf, 6, 12, 18, np.inf],
            labels=['Part Time', 'Half Time', 'Full Time', 'Overload']
        )
    )

def add_standing_features(df):
    """Add class standing-related features."""
    standing_map = {
        'Freshman': 'First Year',
        'Sophomore': 'Second Year',
        'Junior': 'Third Year',
        'Senior': 'Fourth Year'
    }
    
    return df.assign(
        Class_Standing_Self_Reported=df['Class_Standing'].map(standing_map).fillna(df['Class_Standing']),
        Class_Standing_BGSU=pd.cut(
            df['Total_Credit_Hours_Earned'],
            bins=[-np.inf, 30, 60, 90, 120, np.inf],
            labels=['Freshman', 'Sophomore', 'Junior', 'Senior', 'Extended']
        )
    )

def calculate_occupancy(group):
    """Calculate occupancy for a group of check-ins."""
    # Specify format for datetime conversion
    check_in_times = pd.to_datetime(group['Check_In_Time'].astype(str), format='%H:%M:%S')
    check_out_times = pd.to_datetime(group['Check_Out_Time'].astype(str), format='%H:%M:%S')
    
    arrivals = range(1, len(group) + 1)
    departures = [
        sum(1 for j in range(i + 1)
            if not pd.isna(check_out_times.iloc[j]) 
            and check_out_times.iloc[j] <= check_in_times.iloc[i])
        for i in range(len(group))
    ]
    
    return [a - d for a, d in zip(arrivals, departures)]

def add_occupancy(df):
    """Add occupancy calculations to the dataframe."""
    df = df.sort_values(['Check_In_Date', 'Check_In_Time'])
    return df.assign(
        Occupancy=df.groupby('Check_In_Date', group_keys=False)
                    .apply(calculate_occupancy, include_groups=False)
                    .explode()
                    .values
    )

def prepare_dates(df):
    """Convert date and time columns to appropriate formats."""
    return df.assign(
        Check_In_Date=pd.to_datetime(df['Check_In_Date'], format='%m/%d/%y'),
        Check_In_Time=pd.to_datetime(df['Check_In_Time'].astype(str), format='%H:%M:%S').dt.time,
        Check_Out_Time=pd.to_datetime(df['Check_Out_Time'].astype(str), format='%H:%M:%S').dt.time,
    )

def engineer_features(df):
    """Apply all feature engineering transformations."""
    return (df
            .pipe(prepare_dates)
            .pipe(add_temporal_features)
            .pipe(add_time_period)
            .pipe(add_course_features)
            .pipe(add_performance_indicators)
            .pipe(add_session_features)
            .pipe(add_credit_load_features)
            .pipe(add_standing_features)
            .pipe(add_occupancy)
    )

# Process the data
lc_engineered = engineer_features(data_raw)

# -----------------------------------------------------------------------------
# SAVE ENGINEERED DATA
# -----------------------------------------------------------------------------
output_path = Path("data/LC_engineered_py.csv")
lc_engineered.to_csv(output_path, index=False)
