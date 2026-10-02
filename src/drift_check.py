import pandas as pd
import numpy as np
import yaml
import joblib
import os
from sklearn.metrics import mean_squared_error

# Modern / Legacy compatibility for Evidently AI
try:
    from evidently import Report
    from evidently.presets import DataDriftPreset
except ModuleNotFoundError:
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ModuleNotFoundError:
        from evidently.legacy.report import Report
        from evidently.legacy.metric_preset import DataDriftPreset

cols = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f'sensor_{i}' for i in range(1, 22)]

def add_rul(df, cap):
    max_cycle = df.groupby('unit')['cycle'].transform('max')
    rul = max_cycle - df['cycle']
    df['RUL'] = rul.clip(upper=cap)
    return df

def main():
    params = yaml.safe_load(open('params.yaml'))
    
    # 1. Load baseline training dataset (FD001)
    reference_df = pd.read_csv('data/processed/train_FD001_processed.csv')
    
    # 2. Load FD002 dataset
    fd002_path = 'data/raw/train_FD002.txt' if os.path.exists('data/raw/train_FD002.txt') else 'data/raw/test_FD002.txt'
    raw_fd002 = pd.read_csv(fd002_path, sep=r'\s+', header=None, names=cols)
    raw_fd002 = add_rul(raw_fd002, params['rul_cap'])
    
    # Select feature columns matching reference dataset
    feature_cols = [c for c in reference_df.columns if c not in ('unit', 'cycle', 'RUL')]
    
    # Safely scale FD002 features (avoiding divide-by-zero on constant columns)
    current_df = raw_fd002.copy()
    for col in feature_cols:
        min_val = current_df[col].min()
        max_val = current_df[col].max()
        if max_val - min_val > 1e-6:
            current_df[col] = (current_df[col] - min_val) / (max_val - min_val)
        else:
            current_df[col] = 0.0

    # 3. Generate Evidently Data Drift Report
    print("Generating Evidently Data Drift Report...")
    try:
        report = Report(metrics=[DataDriftPreset()])
    except Exception:
        report = Report([DataDriftPreset()])
        
    result = report.run(reference_data=reference_df[feature_cols], current_data=current_df[feature_cols])
    
    # Save report to HTML across different Evidently API versions
    if hasattr(result, 'save_html'):
        result.save_html('drift_report.html')
    elif hasattr(report, 'save_html'):
        report.save_html('drift_report.html')
    elif hasattr(result, 'save'):
        result.save('drift_report.html')
    else:
        report.save('drift_report.html')
        
    print("Successfully saved visual drift report to drift_report.html")
    
    # 4. Compare model RMSE on FD001 vs FD002
    model = joblib.load('model.pkl')
    preds_fd002 = model.predict(current_df[feature_cols])
    rmse_fd002 = np.sqrt(mean_squared_error(current_df['RUL'], preds_fd002))
    
    test_fd001 = pd.read_csv('data/processed/test_split.csv')
    preds_fd001 = model.predict(test_fd001[feature_cols])
    rmse_fd001 = np.sqrt(mean_squared_error(test_fd001['RUL'], preds_fd001))
    
    print("\n================ Model Drift Results ================")
    print(f"FD001 Test RMSE (Baseline):      {rmse_fd001:.2f}")
    print(f"FD002 Test RMSE (Shifted Data):    {rmse_fd002:.2f}")
    print("======================================================")

if __name__ == '__main__':
    main()