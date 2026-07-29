import os, pandas as pd, hopsworks
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("HOPSWORKS_API_KEY")
project = hopsworks.login(project="mlopsaqi123", api_key_value=api_key)
fs = project.get_feature_store()

dummy_df = pd.DataFrame({"city": ["karachi"], "value": [1.0]})

test_fg = fs.get_or_create_feature_group(
    name="dummy_test",
    version=1,
    primary_key=["city"],
    online_enabled=False,
    time_travel_format="HUDI",   # ye line add ki
)

test_fg.insert(dummy_df)
print("✅ Dummy insert worked!")