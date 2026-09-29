from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64

# 1. Point Feast to your generated Parquet dataset
engine_source = FileSource(
    name="engine_feature_source",
    path="data/engine_features.parquet",
    timestamp_field="event_timestamp",
)

# 2. Define the Engine Entity (unit_id)
engine_entity = Entity(
    name="unit_id",
    value_type=ValueType.INT64,
    description="Engine unit identifier",
)

# 3. Define non-constant sensor & operational feature fields
sensor_names = [f'sensor_{i}' for i in [2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]
op_names = ['op1', 'op2', 'op3']
schema_fields = [Field(name=col, dtype=Float32) for col in op_names + sensor_names + ['RUL']]

# 4. Create the Feature View
engine_feature_view = FeatureView(
    name="engine_sensor_features",
    entities=[engine_entity],
    ttl=timedelta(days=365),
    schema=schema_fields,
    online=True,
    source=engine_source,
)