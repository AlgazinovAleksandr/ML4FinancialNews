import os

from dotenv import load_dotenv

# Load .env from the agent/ directory itself
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

API_KEY = os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/devstral-2512:free")
BASE_URL = os.getenv("BASE_URL", "https://openrouter.ai/api/v1")

# Absolute path to the training parquet — resolved relative to this file so the
# agent works regardless of which directory it is invoked from.
DATA_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "data_sample",
        "news_prices_full_processed_with_target_v2.parquet",
    )
)
