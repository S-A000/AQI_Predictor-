import os
import hopsworks
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

if os.name == 'nt':
    os.makedirs('/tmp', exist_ok=True)

def register_cloud_feature_group():
    load_dotenv()
    api_key = os.getenv("HOPSWORKS_API_KEY")
    
    print("🔌 Connecting to Hopsworks Serverless...")
    project = hopsworks.login(project="mlopsaqi123", api_key_value=api_key)
    fs = project.get_feature_store()
    
    # Cloud par jo file upload ki hai, usay direct Feature Group mein map kar rahe hain
    print("🏗️ Creating Feature Group from uploaded cloud file...")
    
    # Hopsworks files path se data read karega
    dataset_api = project.get_dataset_api()
    # User folder ke andar file padi hai
    
    aqi_fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        description="Air Quality features for PK cities",
        primary_key=["city"],
        event_time="timestamp",
        online_enabled=True,
        time_travel_format="DELTA"
    )
    
    # Cloud storage par mojood file se data read kar ke insert karna
    # Since file already Hopsworks dataset mein hai:
    uploaded_file_path = f"hdfs:///Projects/mlopsaqi123/Resources/aqi_features.parquet" # or reading via pandas from dataset api
    
    print("✅ Feature Group metadata registered successfully!")

if __name__ == "__main__":
    register_cloud_feature_group()