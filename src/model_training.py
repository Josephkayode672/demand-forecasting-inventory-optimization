import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from datetime import timedelta
import src.feature_engineering as fe

def train_baseline_model(X_train, y_train):
    scalar = StandardScaler()
    scalar.fit(X_train)
    model = LinearRegression()
    X_train_scaled = scalar.transform(X_train)
    model.fit(X_train_scaled, y_train)
    return model, scalar

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100,
                                  max_depth=10,
                                  random_state=42,
                                  n_jobs=-1
                                  )
    model.fit(X_train, y_train)
    return model

def train_xgboost_model(X_train, y_train):
    model = XGBRegressor(random_state=42,
                          n_estimators=100, 
                          learning_rate=0.1,
                          max_depth=5
                          )
    model.fit(X_train, y_train)
    return model

def hyperparameter_tuning_rf(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(model, param_grid, 
                               cv=TimeSeriesSplit(n_splits=5), 
                               scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    print(f"Best Random forest Paramaters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def hyperparameter_tuning_xgboost(X_train, y_train):
    param_grid = {
        'n_estimators' : [100, 200, 300],
        'max_depth' : [3, 5, 7],
        'learning_rate' : [0.01, 0.05, 0.1],
        'subsample' : [0.7, 0.8, 1.0],
        'colsample_bytree' : [0.7, 0.8, 1.0]
        }
    
    model = XGBRegressor(random_state=42)
    grid_search = GridSearchCV(model, param_grid, 
                               cv=TimeSeriesSplit(n_splits=5), 
                               scoring='neg_mean_squared_error',)
    grid_search.fit(X_train, y_train)
    print(f"Best XGBoost Paramaters: {grid_search.best_params_}")
    return grid_search.best_estimator_


def predict_next_day_sales(model, data):

    #Predict sales for the next day using the trained model and existing feature functions.
    # Get the date for next day
    last_date = data['order_date'].max()
    next_day = last_date + timedelta(days=1)
    
    # Create a Dataframe for nextday prediction    
    next_day_df = pd.DataFrame({'order_date': [next_day]})

    # Concatenate with existing data to ensure feature engineering functions work correctly
    df = pd.concat([data, next_day_df], ignore_index=True)
    
    # Apply features from feature engineering.py module
    df = fe.chronological_features(df)
    df = fe.lag_features(df)  
    df = fe.rolling_features(df)

    next_day_sales = df.drop(['order_date','quantity'], axis=1)
    prediction = model.predict(next_day_sales)[0]
    
    return prediction