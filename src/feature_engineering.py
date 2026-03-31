from numpy import dtype
import pandas as pd
import holidays

# Create chronological features 
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

    years = df['order_date'].dt.year.unique()
    holiday = holidays.US(years= years)
    df['is_holiday'] = df['order_date'].dt.date.isin(holiday).astype(int)
    return df

def lag_features(df, lags=[1,7,14]):

    df = df.copy()
    if 'quantity' not in df.columns:
        raise KeyError("Column 'quantity' not found.")
    
    for lag in lags:
        df[f'sales_lag_{lag}'] = df['quantity'].shift(lag)
    return df

def rolling_features(df):
        
    df = df.copy()
    if 'quantity' not in df.columns:
        raise KeyError("Column 'quantity' not found.")
    
    df['Rolling_mean_7'] = df['quantity'].shift(1).rolling(7).mean()
    df['Rolling_mean_14'] = df['quantity'].shift(1).rolling(14).mean()

    df['Rolling_std_7'] = df['quantity'].shift(1).rolling(7).std()

    df['Rolling_min_7'] = df['quantity'].shift(1).rolling(7).min()
    
    df['Rolling_max_7'] = df['quantity'].shift(1).rolling(7).max()
    return df

def finalize_features(df):

    df = df.dropna()
    df = df.sort_values('order_date').reset_index(drop=True)
    return df

def split_data(df):

    df = df.copy()
    split_date = df['order_date'].max() - pd.DateOffset(months=2)

    train = df[df['order_date'] < split_date]
    test = df[df['order_date'] >= split_date]

    X_train = train.drop(['quantity', 'order_date'], axis =1)
    y_train = train['quantity']

    X_test = test.drop(['quantity', 'order_date'], axis =1)
    y_test = test['quantity']

    return X_train, y_train, X_test, y_test
