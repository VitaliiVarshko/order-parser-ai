# load_to_qdrant.py
"""
Uploading products from CSV to Qdrant using OpenAI-compatible embeddings.
"""

import csv
import multiprocessing
import os
import re
import time
from typing import List, Dict, Any, Optional
import hashlib

#from openai import embeddings
from config import LOG_DIR, LOG_DIR_TOV, LOG_ENABLED_TOV
from datetime import datetime
import json
from config import CSV_PATH

from config import (
    EMBEDDING_PROVIDER,
    EMBEDDING_CONFIG,
    QDRANT_CONFIG
)
from embedding_client import OpenAIEmbeddingClient
from qdrant_manager import QdrantManager

collection_name = QDRANT_CONFIG["collection_name"]

# Длины полей
MAX_LENGTH_NAIM = 150
MAX_LENGTH_CODE = 11
MAX_LENGTH_MARK = 100

# --- Statistics ---
class LoadStats:
    def __init__(self):
        self.total_rows = 0
        self.loaded_count = 0
        self.skipped_count = 0
        self.error_count = 0
        self.duplicate_count = 0
        self.start_time = time.time()
        self.embedding_stats = None
        self.embedding_tokens = 0
        self.updated_count = 0
        self.skipped_count = 0
  
load_stats = LoadStats()


def save_load_log(log_data: Dict[str, Any]):
    if not LOG_ENABLED_TOV:
        return

    try:
        if not os.path.exists(LOG_DIR_TOV):
            os.makedirs(LOG_DIR_TOV)

        timestamp = datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

        filename = f"load_{timestamp}.json"
        filepath = os.path.join(
            LOG_DIR_TOV,
            filename
        )

        with open(
            filepath,
            'w',
            encoding='utf-8'
        ) as f:

            json.dump(
                log_data,
                f,
                ensure_ascii=False,
                indent=2,
                default=str
            )

    except Exception:
    # We don't specifically output anything here.
    # If logging is critical, this error
    # can be handled separately.
        pass


def get_existing_items(qdrant: QdrantManager, collection_name: str) -> Dict[str, Dict]:
    """
    Gets all existing products from Qdrant with their data.
    Returns a dictionary: {code: {"name": ..., "mark": ..., "id": ...}}
    """
    existing = {}
    try:
        # Use scroll to get all the points
        scroll_result = qdrant.client.scroll(
            collection_name=collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )

        N=1
        
        points = scroll_result[0]
        while points:
            for point in points:
                payload = point.payload
                code = payload.get('code', '')
                if code:
                    existing[code] = {
                        "name": payload.get('name', ''),
                        "mark": payload.get('mark', ''),
                        "id": point.id
                    }

                    if N <= 5:  # We display the first 5 for verification
                        # print(N,code)
                        # print(existing[code])
                        N += 1

            # If there is a next page
            if scroll_result[1] is not None:
                scroll_result = qdrant.client.scroll(
                    collection_name=collection_name,
                    limit=10000,
                    offset=scroll_result[1],
                    with_payload=True,
                    with_vectors=False
                )
                points = scroll_result[0]
            else:
                break
                
        # print(f"  Existing products in Qdrant: {len(existing)}")
        return existing
    except Exception as e:
        # print(f"  ⚠️ Failed to retrieve existing items: {e}")
        return {}

# --- Auxiliary functions ---
def clean_string(value, max_length, is_required=False):
    if value is None:
        value = ""
    value = str(value).strip()
    if is_required and not value:
        return None
    value = re.sub(r'[\r\n\t\x00-\x1f\x7f]', ' ', value)
    value = re.sub(r'\s+', ' ', value)
    return value[:max_length].strip()


def generate_numeric_id(code: str, suffix: int = 0) -> int:
    """Generates a numeric ID from a code string"""
    # Use hash to get a number
    hash_val = int(hashlib.md5(code.encode()).hexdigest()[:12], 16)
    # Add suffix for duplicates
    return hash_val + suffix

def create_embedding_client():
    """Creates an embedding client based on the configuration"""
    if EMBEDDING_PROVIDER in EMBEDDING_CONFIG:
        config = EMBEDDING_CONFIG[EMBEDDING_PROVIDER]
        return OpenAIEmbeddingClient(
            api_key=config['api_key'],
            base_url=config['base_url'],
            model=config['model'],
            default_headers=config.get('default_headers', {})
        )
    else:
        raise ValueError(f"Provider '{EMBEDDING_PROVIDER}' not found")

# --- Basic download function ---
def main():
    global load_stats
    load_stats = LoadStats()


    start_time = time.time()
    start_dt = datetime.now().isoformat()
    
    # --- Структура для лога ---
    log_data = {
        "timestamp_start": start_dt,
        "source_file": CSV_PATH,
        "collection_name": collection_name,
        "embedding_provider": EMBEDDING_PROVIDER,
        "embedding_model": EMBEDDING_CONFIG.get(EMBEDDING_PROVIDER, {}).get('model', 'unknown'),
        "status": "started",
        "errors": [],
        "stats": {},
        "timestamp_end": None,
        "duration_seconds": None
    }

    try:
      
        # 5. Reading CSV and preparing points
        print(f"\nߓ Reading a file: {CSV_PATH}")
              
        if not os.path.exists(CSV_PATH):
            print(f"❌ Error: file not found: {CSV_PATH}")
            return
      
          
        # 3. Initializing Qdrant
        qdrant = QdrantManager(
            host=QDRANT_CONFIG["host"],
            port=QDRANT_CONFIG["port"]
            )

        # 4. Creating a collection
                   
        # --- Getting existing items from Qdrant ---
        existing_items = get_existing_items(qdrant, collection_name)


        points_to_add = []     # New products
        points_to_update = []  # Products with changes
        points_to_skip = 0     # Products without changes
        duplicate_counter = {}
        
        with open(CSV_PATH, 'r', encoding='utf-8-sig') as csvfile:
            header_line = csvfile.readline().strip()
            header = header_line.split(';;')
     
            if 'Наименование' not in header or 'Код' not in header:
                print("❌ Error: Required columns 'Наименование' or 'Код' are missing")
                return
            
            idx_name = header.index('Наименование')
            idx_code = header.index('Код')
            idx_mark = header.index('Марка') if 'Марка' in header else -1
            
            # We read the lines
            for line in csvfile:
                line = line.strip()
                if not line:
                    continue
                
                load_stats.total_rows += 1
                
                fields = line.split(';;')
                if len(fields) < 3:
                    load_stats.error_count += 1
                    continue
                
                naim = clean_string(fields[idx_name] if idx_name < len(fields) else "", MAX_LENGTH_NAIM, is_required=True)
                if naim is None:
                    load_stats.skipped_count += 1
                    continue
                
                code = clean_string(fields[idx_code] if idx_code < len(fields) else "", MAX_LENGTH_CODE, is_required=True)
                if code is None:
                    load_stats.skipped_count += 1
                    continue
                
                mark = ""
                if idx_mark != -1 and idx_mark < len(fields):
                    mark = clean_string(fields[idx_mark], MAX_LENGTH_MARK, is_required=False)
                
                # Processing duplicates
                if code in duplicate_counter:
                    duplicate_counter[code] += 1
                    doc_id = generate_numeric_id(code, duplicate_counter[code])
                    load_stats.duplicate_count += 1
                else:
                    duplicate_counter[code] = 0
                    doc_id = generate_numeric_id(code)

                naimlower=naim.lower() # Для векторизации используем только наименование, переводим в нижний регистр

    
                # --- Upsert logics ---
                if code in existing_items:
                    existing = existing_items[code]
                    # Checking if the data has changed
                    if existing['name'] == naimlower and existing['mark'] == mark:
                        points_to_skip += 1
                    else:
                        # The data has changed - we are updating it
                        points_to_update.append({
                            "id": existing['id'],  # Use the existing ID
                            "code": code,
                            "name": naimlower,
                            "mark": mark,
                            "text": naimlower,
                            "action": "update"
                        })
                else:
                    # New product
                    points_to_add.append({
                        "id": doc_id,
                        "code": code,
                        "name": naimlower,
                        "mark": mark,
                        "text": naimlower,
                        "action": "add"
                    })



        # --- Connecting the dots to load ---
        points_to_load = points_to_add + points_to_update
    
        log_data["total_rows_read"] = load_stats.total_rows
        log_data["points_to_load"] = len(points_to_load)
        log_data["duplicates_found"] = load_stats.duplicate_count
        log_data["errors_read"] = load_stats.error_count
        log_data["skipped"] = load_stats.skipped_count

        if not points_to_load:
            print("\n✅ All data is current, no updates required")
            # load_stats.print_stats()
            return



        # 1. Creating an embedding client
        embedding_client = create_embedding_client()
        
        # 2. We get the vector dimension (we make a test query)
        test_embedding = embedding_client.embed_single("тест")
        if not test_embedding:
            print("❌ Error: Failed to get test embedding")
            return
        
        vector_size = len(test_embedding)
   
        try:
            qdrant.client.get_collections()
            print("✓ Qdrant is running and responding")
        except Exception as e:
            raise Exception(f"Qdrant is not responding: {e}")
        

        # 6. We receive embeddings only for modified products ---
        print(f"\nߚ Getting embeddings for {len(points_to_load)} products...")

        texts = [p["text"] for p in points_to_load]

        start_time2 = time.time()

        embeddings = embedding_client.embed(
            texts,
            batch_size=128
        )

        embed_elapsed = time.time() - start_time2

        # CRITICAL CHECK
        if len(embeddings) != len(points_to_load):

            raise RuntimeError(
                f"The number of embeddings does not match "
                f"with the number of goods: "
                f"expected {len(points_to_load)}, "
                f"received {len(embeddings)}"
            )


        # ---------------------------------------------------------
        # Only now can you change the Qdrant collection.
        # All embeddings successfully retrieved.
        # ---------------------------------------------------------

        elapsed = time.time() - start_time
        print(f"  Embeddings received: {len(embeddings)} in {elapsed:.2f} sec")
            
        # 7. Forming points for Qdrant
        qdrant_points = []
        for i, point in enumerate(points_to_load):
            if i >= len(embeddings):
                break
            
            qdrant_points.append({
                "id": point["id"],
                "vector": embeddings[i],
                "payload": {
                    "code": point["code"],
                    "name": point["name"],
                    "mark": point["mark"]
                }
            })
        
        # 8. Uploading to Qdrant
        qdrant_start = time.time()
        print(f"\nߚ Uploading to Qdrant...")
        qdrant.add_points(collection_name, qdrant_points, batch_size=100)
        qdrant_elapsed = time.time() - qdrant_start
        
        load_stats.loaded_count = len(qdrant_points)
        load_stats.skipped_count = (points_to_skip)

        log_data["qdrant_load_time"] = round(qdrant_elapsed, 2)
        log_data["points_loaded"] = load_stats.loaded_count
        
        # 9. Verification
        info = qdrant.get_collection_info(collection_name)
        print(f"\n--- Verification ---")
        print(f"  Name: {info['name']}")
        print(f"  Number of points: {info['points_count']}")
        print(f"  Status: {info['status']}")

        expected_points = len(points_to_load)
        actual_points = len(qdrant_points)

        log_data["collection_after_load"] = {
            "points_count": actual_points,
            "expected_points": expected_points,
            "status": info.get('status', 'unknown')
        }

        if actual_points != expected_points:
            raise RuntimeError(
                f"Qdrant check error: "
                f"expected {expected_points} points, "
                f"received {actual_points}"
            )

        if info.get('status') != 'green':
            raise RuntimeError(
                f"Qdrant collection has an unexpected status: "
                f"{info.get('status')}"
            )

        load_stats.loaded_count = actual_points
        log_data["points_loaded"] = actual_points

        log_data["status"] = "success"

 
    except Exception as e:
        error_msg = str(e)

        log_data["status"] = "error"

        log_data["errors"].append({
            "type": type(e).__name__,
            "message": error_msg,
            "timestamp": datetime.now().isoformat()
        })
    
    finally:
        elapsed = time.time() - start_time

        log_data["timestamp_end"] = datetime.now().isoformat()
        log_data["duration_seconds"] = round(elapsed, 3)

        log_data["stats"] = {
            "total_rows": load_stats.total_rows,
            "loaded_count": load_stats.loaded_count,
            "skipped_count": load_stats.skipped_count,
            "error_count": load_stats.error_count,
            "duplicate_count": load_stats.duplicate_count
        }

        # If embedding_client exists,
        # we store its errors even if an exception occurs.
        if 'embedding_client' in locals():

            embed_stats = embedding_client.get_stats()

            log_data["embedding"] = {
                "calls": embed_stats.get(
                    "total_requests",
                    0
                ),
                "tokens": embed_stats.get(
                    "total_tokens",
                    0
                ),
                "time_seconds": (
                    embed_elapsed
                    if 'embed_elapsed' in locals()
                    else 0
                ),
                "model": embed_stats.get(
                    "model",
                    "unknown"
                ),
                "provider": embed_stats.get(
                    "provider",
                    "unknown"
                ),
                "errors": embed_stats.get(
                    "errors",
                    []
                ),
                "request_history": embed_stats.get(
                    "request_history",
                    []
                )
            }


        if 'qdrant' in locals():

            log_data["qdrant"] = {
                "errors": qdrant.errors,
                "upsert_history": qdrant.upsert_history
            }

        save_load_log(log_data)
        
        

if __name__ == "__main__":
    multiprocessing.freeze_support()  
    main()