from src.data_processing import *

def main():

    data = read_file('data/Pizza Sales.csv')

    data = convert_date(data)

    data = aggregate_daily_sales(data)

    data = fill_missing_data(data)

    save_cleaned_csv(data, 'data/Cleaned_Pizza_Sales.csv')

    plot_all(data)

main()