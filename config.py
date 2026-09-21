# config.py
"""
Project configuration. All values ​​are read from the .env file..
"""

import os
from dotenv import load_dotenv

# Loading variables from .env
load_dotenv()

# --- Mistral API ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

# --- Qdrant ---
QDRANT_CONFIG = {
    "host": os.getenv("QDRANT_HOST", "localhost"),
    "port": int(os.getenv("QDRANT_PORT", "6333")),
    "collection_name": os.getenv("QDRANT_COLLECTION", "nomenclature")
}

# --- Paths ---
CSV_PATH = r"D:\TovAI\tov.csv"
LOG_DIR = r"D:\TovAI\1C_Logs"

# --- Logging ---
LOG_ENABLED = True
LOG_ENABLED_TOV = True
LOG_DIR_TOV =  r"D:\TovAI\Tov_Logs"

# --- Search ---
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
SEARCH_RESULTS = int(os.getenv("SEARCH_RESULTS", 10))

# --- Providers ---
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "mistral")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mistral")

# --- Embedding Configuration ---
EMBEDDING_CONFIG = {
    "mistral": {
        "api_key": MISTRAL_API_KEY,
        "base_url": "https://api.mistral.ai/v1",
        "model": "mistral-embed",
        "default_headers": {}
    },
    "openai": {
        "api_key": "",
        "base_url": "https://api.openai.com/v1",
        "model": "text-embedding-3-small",
        "default_headers": {}
    },
    "deepseek": {
        "api_key": "",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-embed",
        "default_headers": {}
    }
}

# --- LLM Configuration ---
LLM_CONFIG = {
    "mistral": {
        "api_key": MISTRAL_API_KEY,
        "base_url": "https://api.mistral.ai/v1",
        "model": "mistral-medium-latest",
        "default_headers": {}
    },
    "openai": {
        "api_key": "",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "default_headers": {}
    },
    "openrouter": {
        "api_key": "",  
        "base_url": "https://openrouter.ai/api/v1",  
         "model": "minimax/minimax-m2.7:free", 
        "default_headers": {
            "HTTP-Referer": "http://localhost:8000",  # optional, for statistics
            "X-Title": "OrderParser"  # optional, name of your application
        }
    }
}