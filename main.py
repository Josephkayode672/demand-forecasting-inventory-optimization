from src.data_processing import read_file, convert_date, aggregate_daily_sales,\
fill_missing_data, save_cleaned_csv, plot_all

def main():
    # Main pipeline for data processing and EDA.

    # Steps 1: load raw CSV dataset into a pandas dataframe
    data = read_file('data/Pizza Sales.csv')

    # Step 2: convert 'order_date' column to datetime and sort values
    data = convert_date(data)

    # Step 3: aggregate total quantity sold per day
    data = aggregate_daily_sales(data)

    # Step 4: fill missing date with zero sales for time series consistency
    data = fill_missing_data(data)

    # step 5: save cleaned and processed dataset into for reuse later
    save_cleaned_csv(data, 'data/Cleaned_Pizza_Sales.csv')

    #step 6: generate all EDA plots(trend, pattern, seasonality, ACF, PACF)
    plot_all(data)

if __name__ == "__main__":
    main()