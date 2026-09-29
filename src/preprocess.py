import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import MinMaxScaler

cols = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f'sensor_{i}' for i in range(1, 22)]

def load_raw(path):
    return pd.read_csv(path, sep=r'\s+', header=None, names=cols)

def add_rul(df, cap):
    max_cycle = df.groupby('unit')['cycle'].transform('max')
    rul = max_cycle - df['cycle']
    df['RUL'] = rul.clip(upper=cap)
    return df

def main():
    params = yaml.safe_load(open('params.yaml'))
    df = load_raw('data/raw/train_FD001.txt')
    df = add_rul(df, params['rul_cap'])

    # Drop near-zero-variance sensors (constant across all engines)
    sensor_cols = [c for c in df.columns if c.startswith('sensor_')]
    keep = [c for c in sensor_cols if df[c].std() > 1e-6]
    dropped = [c for c in sensor_cols if c not in keep]
    print('Dropping constant sensors:', dropped)

    feature_cols = ['op1', 'op2', 'op3'] + keep
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    df.to_csv('data/processed/train_FD001_processed.csv', index=False)
    print('Saved processed data:', df.shape)

if __name__ == '__main__':
    main()