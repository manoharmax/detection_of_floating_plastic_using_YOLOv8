from dotenv import load_dotenv
import os

load_dotenv()  # This loads the .env file
api_key = os.getenv("PEXELS_API_KEY")
print("API Key: - test_env_load.py:6", api_key)
