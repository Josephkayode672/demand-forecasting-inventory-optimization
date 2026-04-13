import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,root_mean_squared_error, r2_score


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    print(f"Model Evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return preds, mae, rmse

def feature_importance(model, X_train):
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importance
    }).sort_values(by='importance', ascending=False)
    return importance_df

def plot_feature_importance(importance_df):
    plt.figure(figsize=(10,6))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.show()

def plot_combined(y_test, preds):
    plt.figure(figsize=(14,6))

    # -------- Plot 1: Time Series --------
    plt.subplot(1, 2, 1)
    plt.plot(y_test.values, label='Actual Sales')
    plt.plot(preds, label='Predicted Sales')
    plt.title('Actual vs Predicted (Time Series)')
    plt.xlabel('Time')
    plt.ylabel('Sales')
    plt.legend()

    # -------- Plot 2: Scatter --------
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, preds, alpha=0.5, label='Predictions')
    
    # Perfect prediction line
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', label='Perfect Prediction')
    
    plt.title('Actual vs Predicted (Scatter)')
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.legend()

    plt.tight_layout()
    plt.show()
