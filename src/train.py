import pandas as pd
import yaml
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit

def main():
    params = yaml.safe_load(open('params.yaml'))
    df = pd.read_csv('data/processed/train_FD001_processed.csv')

    feature_cols = [c for c in df.columns if c not in ('unit', 'cycle', 'RUL')]
    X, y, groups = df[feature_cols], df['RUL'], df['unit']

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=params['test_size'], random_state=params['random_state'])
    train_idx, test_idx = next(splitter.split(X, y, groups))

    model = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        random_state=params['random_state'])
    model.fit(X.iloc[train_idx], y.iloc[train_idx])

    joblib.dump(model, 'model.pkl')
    X.iloc[test_idx].assign(RUL=y.iloc[test_idx]).to_csv('data/processed/test_split.csv', index=False)

if __name__ == '__main__':
    main()