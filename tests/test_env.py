from dotenv import load_dotenv
import os

load_dotenv()

print(os.getenv("AQICN_API_KEY"))
print(os.getenv("OPENWEATHER_API_KEY"))
print(os.getenv("LOCATION_CITY"))
print(os.getenv("LOCATION_COUNTRY"))
print(os.getenv("PATHS_RAW_DATA"))