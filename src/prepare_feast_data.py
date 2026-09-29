import pandas as pd
import os

def main():
    # Ensure the feature_repo/data folder exists
    os.makedirs('feature_repo/data', exist_ok=True)

    # Load the processed data from Step 2
    df = pd.read_csv('data/processed/train_FD001_processed.csv')

    # Rename or cast 'unit' to 'unit_id' for Feast entity tracking
    df['unit_id'] = df['unit'].astype('int64')

    # Feast requires an event timestamp column for temporal queries
    df['event_timestamp'] = pd.date_range(start='2026-01-01', periods=len(df), freq='h')

    # Save to Parquet format expected by Feast
    output_path = 'feature_repo/data/engine_features.parquet'
    df.to_parquet(output_path, index=False)
    print(f"Successfully created Feast dataset at {output_path} with shape {df.shape}")

if __name__ == '__main__':
    main()