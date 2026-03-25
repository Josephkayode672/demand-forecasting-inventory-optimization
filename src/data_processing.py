import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def read_file(path):
    """
    Reads a CSV file into a pandas DataFrame.
    
    Args:
        path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        pd.errors.EmptyDataError: If the file is empty.
    """
    try:
        df = pd.read_csv(path)
        if df.empty:
            raise pd.errors.EmptyDataError("The CSV file is empty.")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {path}")
    except Exception as e:
        raise Exception(f"Error reading file: {e}")

def convert_date(df):
    """
    Converts 'order_date' column to datetime and sorts the DataFrame by date.
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'order_date' column.
    
    Returns:
        pd.DataFrame: DataFrame with converted and sorted dates.
    
    Raises:
        KeyError: If 'order_date' column is missing.
        ValueError: If date conversion fails.
    """
    df = df.copy()
    if 'order_date' not in df.columns:
        raise KeyError("Column 'order_date' not found in DataFrame.")
    try:
        df['order_date'] = pd.to_datetime(df['order_date'], errors='raise')
        df = df.sort_values('order_date')
        return df
    except ValueError as e:
        raise ValueError(f"Date conversion failed: {e}")

def aggregate_daily_sales(df):
    """
    Aggregates daily sales by summing 'quantity' per 'order_date'.
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'order_date' and 'quantity'.
    
    Returns:
        pd.DataFrame: DataFrame with daily aggregated sales.
    """
    df = df.copy()
    daily_sales = df.groupby('order_date')['quantity'].sum().reset_index()
    return daily_sales

def fill_missing_data(df):
    """
    Fills missing dates to ensure continuous daily data, filling quantities with 0.
    
    Args:
        df (pd.DataFrame): DataFrame with 'order_date' and 'quantity'.
    
    Returns:
        pd.DataFrame: DataFrame with filled missing dates.
    """
    df = df.copy()
    df = df.set_index('order_date')
    df = df.asfreq('D')  # Daily frequency
    df['quantity'] = df['quantity'].fillna(0)
    df = df.reset_index()
    return df

def save_cleaned_csv(df, path):
    """
    Save cleaned DataFrame to CSV.
    Args:
        df (pd.DataFrame): cleaned dataset
        path (str): output file path
    """
    df.to_csv(path, index=False)
    print(f"Saved cleaned data to: {path}")


def plot_sales(df):
    """
    Plots overall daily sales trend.
    
    Args:
        df (pd.DataFrame): DataFrame with 'order_date' and 'quantity'.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df['order_date'], df['quantity'])
    plt.title('Retail Daily Sales Trend')
    plt.xlabel('Date')
    plt.ylabel('Sales Quantity')
    plt.grid(True)
    plt.show()

def plot_weekly_pattern(df):
    """
    Plots average sales by day of the week.
    
    Args:
        df (pd.DataFrame): DataFrame with 'order_date' and 'quantity'.
    """
    df = df.copy()
    df['day_of_week'] = df['order_date'].dt.day_name()
    weekly_avg = df.groupby('day_of_week')['quantity'].mean()
    weekly_avg = weekly_avg.reindex(['Monday', 'Tuesday', 'Wednesday',
                                      'Thursday', 'Friday', 'Saturday', 'Sunday'])
    weekly_avg.plot(kind='bar', title='Average Sales per Day of Week')
    plt.xticks(rotation=45)
    plt.ylabel('Average Quantity')
    plt.grid(True)
    plt.show()

def plot_monthly_pattern(df):
    """
    Plots average sales by month.
    
    Args:
        df (pd.DataFrame): DataFrame with 'order_date' and 'quantity'.
    """
    df = df.copy()
    df['month'] = df['order_date'].dt.month_name()
    monthly_avg = df.groupby('month')['quantity'].mean()
    monthly_avg = monthly_avg.reindex(['January', 'February', 'March', 'April', 'May', 'June',
                                       'July', 'August', 'September', 'October', 'November', 'December'])
    monthly_avg.plot(kind='bar', title='Average Sales per Month')
    plt.xticks(rotation=45)
    plt.ylabel('Average Quantity')
    plt.grid(True)
    plt.show()

def decompose_sales_weekly(df):
    """
    Performs seasonal decomposition on daily sales with weekly seasonality.
    
    Args:
        df (pd.DataFrame): DataFrame with 'order_date' and 'quantity'.
    """
    daily_sales = df.groupby('order_date')['quantity'].sum()
    daily_sales.index = pd.to_datetime(daily_sales.index)
    daily_sales = daily_sales.asfreq('D').ffill()
    decomposition = seasonal_decompose(daily_sales, model='additive', period=7)
    decomposition.plot()
    plt.show()

def decompose_sales_monthly(df):
    """
    Performs seasonal decomposition on daily sales with monthly seasonality.
    
    Args:
        df (pd.DataFrame): DataFrame with 'order_date' and 'quantity'.
    """
    daily_sales = df.groupby('order_date')['quantity'].sum()
    daily_sales.index = pd.to_datetime(daily_sales.index)
    daily_sales = daily_sales.asfreq('D').ffill()
    decomposition = seasonal_decompose(daily_sales, model='additive', period=30)
    decomposition.plot()
    plt.show()

def plot_ACF(df):
    """
    Plots Autocorrelation Function (ACF) for daily sales.
    
    Args:
        df (pd.DataFrame): DataFrame with 'order_date' and 'quantity'.
    """
    daily_sales = df.groupby('order_date')['quantity'].sum()
    daily_sales.index = pd.to_datetime(daily_sales.index)
    daily_sales = daily_sales.asfreq('D').ffill()
    plot_acf(daily_sales, lags=30)
    plt.title('Autocorrelation (ACF)')
    plt.show()

def plot_PACF(df):
    """
    Plots Partial Autocorrelation Function (PACF) for daily sales.
    
    Args:
        df (pd.DataFrame): DataFrame with 'order_date' and 'quantity'.
    """
    daily_sales = df.groupby('order_date')['quantity'].sum()
    daily_sales.index = pd.to_datetime(daily_sales.index)
    daily_sales = daily_sales.asfreq('D').ffill()
    plot_pacf(daily_sales, lags=30)
    plt.title('Partial Autocorrelation (PACF)')
    plt.show()

def plot_all(df):
    """
    Generates all plots sequentially.
    
    Args:
        df (pd.DataFrame): DataFrame with 'order_date' and 'quantity'.
    """
    plot_sales(df)
    plot_weekly_pattern(df)
    plot_monthly_pattern(df)
    decompose_sales_weekly(df)
    decompose_sales_monthly(df)
    plot_ACF(df)
    plot_PACF(df)