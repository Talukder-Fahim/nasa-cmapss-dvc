import pandas as pd, joblib, json
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def main():
    model = joblib.load('model.pkl')
    test = pd.read_csv('data/processed/test_split.csv')
    feature_cols = [c for c in test.columns if c != 'RUL']

    preds = model.predict(test[feature_cols])
    
    # Calculate MSE and take the square root for RMSE
    mse = mean_squared_error(test['RUL'], preds)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(test['RUL'], preds))

    json.dump({'rmse': rmse, 'mae': mae}, open('metrics.json', 'w'), indent=2)
    print(f'RMSE: {rmse:.2f}  MAE: {mae:.2f}')

if __name__ == '__main__':
    main()