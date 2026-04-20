from src.data_processing import read_file, convert_date, aggregate_daily_sales,\
fill_missing_data, save_cleaned_csv, plot_all
from src.feature_engineering import chronological_features, lag_features, rolling_features,\
    finalize_features, split_data
from src.model_training import train_baseline_model, train_random_forest, train_xgboost_model,\
    hyperparameter_tuning_rf, hyperparameter_tuning_xgboost, predict_next_day_sales
from src.model_evaluation import evaluate_model, feature_importance, plot_feature_importance,\
    plot_combined
import pandas as pd

def main():
    """Main pipeline for data preprocessing and EDA."""

    #======== WEEK: 1 ========
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
    print()

    # step 6: generate all EDA plots(trend, pattern, seasonality, ACF, PACF)
    print("="*40)
    print("WEEK 1: EXPLORATORY DATA ANALYSIS")
    print("="*40)
    plot_all(data)
    print('All EDA ploted successfully')
    print()

    """Main pipeline for Advanced feature Engineering."""

    # ======== WEEK: 2 ========
    # step 7: create chronological features, 
    data = chronological_features(data)

    # step 8: create lag features for quantity sold
    data = lag_features(data)

    # step 9: create rolling mean features for quantity sold
    data = rolling_features(data)

    # step 10: finalize features by dropping rows with NaN values and sorting by date
    data = finalize_features(data)

    # step 11: split data into training and testing sets based on date
    X_train, y_train, X_test, y_test = split_data(data)

    """Main pipeline for Model Training, Selection, and Evaluation."""

    # ======== WEEK: 3 & 4 ========  
    # step 12: train baseline linear regression model and evaluate performance  

    LR_model, scalar = train_baseline_model(X_train, y_train)
    print("Baseline Linear Regression model trained.")
    scaled_X_test = scalar.transform(X_test)
    preds_LR, mae_LR, rmse_LR = evaluate_model(LR_model, scaled_X_test, y_test)
    print()
    
    # step 13: train Random Forest, evaluate performance, and perform hyperparameter tuning
    RF_model = train_random_forest(X_train, y_train)
    print("Random Forest model trained.")
    preds_RF, mae_RF, rmse_RF = evaluate_model(RF_model, X_test, y_test)
    print()
    tune_rf = hyperparameter_tuning_rf(X_train, y_train)
    preds_RF_tuned, mae_RF_tuned, rmse_RF_tuned = evaluate_model(tune_rf, X_test, y_test)
    print()

    # Train XGBoost model, evaluate performance, and perform hyperparameter tuning
    XG_model = train_xgboost_model(X_train, y_train)
    print("XGBoost model trained.")
    preds_XG, mae_XG, rmse_XG = evaluate_model(XG_model, X_test, y_test)
    print()
    tune_xg = hyperparameter_tuning_xgboost(X_train, y_train)
    preds_XG_tuned, mae_XG_tuned, rmse_XG_tuned = evaluate_model(tune_xg, X_test, y_test)
    print()

    Models = {
        'Linear Regression': [mae_LR, rmse_LR],
        'Random Forest': [mae_RF, rmse_RF],
        'Tuned Random Forest': [mae_RF_tuned, rmse_RF_tuned],
        'XGBoost': [mae_XG, rmse_XG],
        'Tuned XGBoost':[mae_XG_tuned, rmse_XG_tuned]
        }

    min_model = min(Models, key=lambda k: Models[k][1])
    print(min_model)
    print()

    #select the best model
    best_model = None
    if min_model == 'Linear Regression':
        best_model = LR_model
    elif min_model == 'Random Forest':
        best_model = RF_model
    elif min_model == 'Tuned Random Forest':
        best_model = tune_rf
    elif min_model == 'XGBoost':
        best_model = XG_model
    elif min_model == 'Tuned XGBoost':
        best_model = tune_xg
    
    importances_df = feature_importance(best_model, X_train)
    importances_df = importances_df.sort_values(by='importance', ascending=False)
    print("Top 10 Feature Importances:")
    print(importances_df.head(10))

    plot_feature_importance(importances_df)
    plot_combined(y_test, preds_XG_tuned)

    # ----- PREDICTION SECTION ------
    print("\n" + "="*40)
    print(' '*10 + 'NEXT DAY SALES PREDICTION')
    print("="*40)
    
    # Make prediction
    prediction_result = predict_next_day_sales(best_model, data)

    print(f'the next day sales quantity is :{prediction_result:.2f}')
    
if __name__ == "__main__":
    main()