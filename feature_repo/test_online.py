from feast import FeatureStore
import pandas as pd

# 1. Initialize FeatureStore from the current repository
store = FeatureStore(repo_path=".")

# 2. Define entity key to query (Engine Unit ID 1)
entity_rows = [{"unit_id": 1}]

# 3. Specify features to retrieve from the feature view
features_to_fetch = [
    "engine_sensor_features:op1",
    "engine_sensor_features:sensor_2",
    "engine_sensor_features:sensor_3",
    "engine_sensor_features:RUL"
]

# 4. Fetch online features
response = store.get_online_features(
    features=features_to_fetch,
    entity_rows=entity_rows
).to_dict()

# Display formatted result
print("--- Feast Online Feature Retrieval Output ---")
print(pd.DataFrame.from_dict(response))