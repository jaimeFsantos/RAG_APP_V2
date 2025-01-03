import os

BASE_DIR = "/home/ec2-user/car-app"
DATA_DIR = os.path.join(BASE_DIR, "data")
SOURCES_DIR = os.path.join(DATA_DIR, "sources")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(DATA_DIR, "models")
TEMP_DIR = os.path.join(DATA_DIR, "temp")

# Create directories if they don't exist
for dir_path in [DATA_DIR, SOURCES_DIR, PROCESSED_DIR, MODEL_DIR, TEMP_DIR]:
    os.makedirs(dir_path, exist_ok=True)