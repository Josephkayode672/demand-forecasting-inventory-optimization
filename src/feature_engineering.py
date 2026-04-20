import pandas as pd
import holidays

# ----Create chronological features----
def chronological_features(df):

    df = df.copy()
    if 'order_date' not in df.columns:
        raise KeyError("Column 'order_date' not found.")
         
    if not pd.api.types.is_datetime64_any_dtype(df['order_date']):
        df['order_date'] = pd.to_datetime(df['order_date'])

    df['day_of_week'] = df['order_date'].dt.dayofweek
    df['month'] = df['order_date'].dt.month
    df['day_of_month'] = df['order_date'].dt.day
    df['week_of_year'] = df['order_date'].dt.isocalendar().week
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    #-----Create holiday features----
    years = df['order_date'].dt.year.unique()
    holiday = holidays.Nigeria(years= years)
    df['is_holiday'] = df['order_date'].dt.date.isin(holiday).astype(int)
    return df

#----Create lag features----
def lag_features(df, lags=[1,7,14,21,28]):

    df = df.copy()
    if 'quantity' not in df.columns:
        raise KeyError("Column 'quantity' not found.")
    
    for lag in lags:
        df[f'sales_lag_{lag}'] = df['quantity'].shift(lag)
    return df

#----Create rolling feature----
def rolling_features(df, windows=[7,14]):
        
    df = df.copy()
    if 'quantity' not in df.columns:
        raise KeyError("Column 'quantity' not found.")
    
    for window in windows:
        df[f'rolling_mean_{window}'] = df['quantity'].shift(1).rolling(window).mean()
        df[f'rolling_std_{window}'] = df['quantity'].shift(1).rolling(window).std()
        df[f'rolling_min_{window}'] = df['quantity'].shift(1).rolling(window).min()
        df[f'rolling_max_{window}'] = df['quantity'].shift(1).rolling(window).max()
        df[f'rolling_median_{window}'] = df['quantity'].shift(1).rolling(window).median()
    return df

# ----drop rolls NAN values----
# ----sort by 'order_date' and reset index----
def finalize_features(df):

    df = df.dropna()
    df = df.sort_values('order_date').reset_index(drop=True)
    return df

def split_data(df):
    """
    Split data using time-based split.
    Uses 80-20 split on entire dataset to ensure adequate test size.
    """
    df = df.copy()
    
    # Use 80% for training and 20% for testing
    split_idx = int(len(df) * 0.8)
    split_date = df.iloc[split_idx]['order_date']
    
    train = df[df['order_date'] < split_date]
    test = df[df['order_date'] >= split_date]
    
    print(f"Train period: {train['order_date'].min()} to {train['order_date'].max()}")
    print(f"Test period: {test['order_date'].min()} to {test['order_date'].max()}")
    print(f"Train size: {len(train)}, Test size: {len(test)}\n")
    
    X_train = train.drop(['quantity', 'order_date'], axis=1)
    y_train = train['quantity']
    
    X_test = test.drop(['quantity', 'order_date'], axis=1)
    y_test = test['quantity']
    
    return X_train, y_train, X_test, y_test
