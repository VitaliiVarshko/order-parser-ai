# order_parser.py
"""
A single entry point for calls from ERP.
Accepts raw order text and returns JSON with a list of products.
"""

import json
import time
import sys
import os

from datetime import datetime
from typing import Dict, Any, List
from config import LOG_DIR, LOG_ENABLED

# Add the path to the project (if necessary)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import LLM_PROVIDER, LLM_CONFIG

from search_qdrant import parse_items, stats, search_order_with_stats


def save_log(log_data: Dict[str, Any]):
    """Saves the log to a JSON file"""
    if not LOG_ENABLED:
        return
    
    # Create a folder for logs if it doesn't exist.
    if not os.path.exists(LOG_DIR):
        try:
            os.makedirs(LOG_DIR)
        except:
            return
    
    # File name: YYYY-MM-DD_HH-MM-SS_user.json
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    user = log_data.get('user', 'unknown')
    filename = f"{timestamp}_{user}.json"
    filepath = os.path.join(LOG_DIR, filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        # If you couldn't save the log, just ignore it.
        pass

def parse_order(text: str, user: str = None) -> str:
    """
    Main function for calls from ERP.
    
   Input: raw order text (multi-line)
    Output: JSON string with an array of products

    Output structure:
    [
        {
            "original": "Масло сливочное 82% - 3 шт",
            "name": "Масло сливочное 82%",
            "qty": "3"
        },
        ...
    ]
    """

    start_time = time.time()
    start_dt = datetime.now().isoformat()

    # --- Structure for log ---
    log_data = {
        "timestamp_start": start_dt,
        "user": user,
        "input_text": text,
        "llm": {},
        "embedding": {},
        "result": {
            "items": [],
            "count": 0
        },
        "errors": [],
        "timestamp_end": None,
        "duration_seconds": None
    }


    try:
        # We reset the statistics for the sake of measurement purity
        stats.llm_call_count = 0
        stats.llm_total_tokens = 0
        stats.llm_total_time = 0.0
        
        # Parsing text using LLM
        items = parse_items(text)

        # Let's check if there is an error
        if stats.last_error:
            # Returning an error to ERP
            error_result = {
                "error": stats.last_error,
                "items": []
            }
            return json.dumps(error_result, ensure_ascii=False)


        # Saving LLM statistics
        log_data["llm"] = {
            "calls": stats.llm_call_count,
            "tokens": stats.llm_total_tokens,
            "time_seconds": stats.llm_total_time,
            "model": LLM_CONFIG.get(LLM_PROVIDER, {}).get('model', 'unknown')
        } if stats.llm_call_count > 0 else {}

        
        # Generating output JSON
        
        result_items = []
        for item in items:
            result_items.append({
                "original": item.get("original", ""),
                "name": item.get("name", ""),
                "qty": item.get("qty", "1")
            })

        log_data["result"]["items"] = result_items
        log_data["result"]["count"] = len(result_items)
        
        # --- 4. We save embedding statistics ---
        log_data["embedding"] = {
            "calls": stats.api_call_count,
            "tokens": stats.api_total_tokens,
            "time_seconds": stats.api_total_time,
            "model": stats.api_model
        } if stats.api_call_count > 0 else {}
        
        # --- 5. We mark the completion ---
        end_time = time.time()
        log_data["timestamp_end"] = datetime.now().isoformat()
        log_data["duration_seconds"] = round(end_time - start_time, 3)
        
        # --- 6. We save the log ---
        save_log(log_data)

        
        return json.dumps(result_items, ensure_ascii=False)
        
    except Exception as e:
         # --- Error handling ---
        error_msg = str(e)
        log_data["errors"].append({
            "type": type(e).__name__,
            "message": error_msg,
            "timestamp": datetime.now().isoformat()
        })
        
        # We record the completion
        log_data["timestamp_end"] = datetime.now().isoformat()
        log_data["duration_seconds"] = round(time.time() - start_time, 3)
        
        # We save the log with the error
        save_log(log_data)
        
        # We return the error to ERP
        error_result = {
            "error": error_msg,
            "items": []
        }
        return json.dumps(error_result, ensure_ascii=False)

def parse_order_with_stats(text: str, user: str = None) -> str:
    """
    Extended version with statistics output to stderr.
    """
    start_time = time.time()
    
    result_json = parse_order(text, user)
    
    elapsed = time.time() - start_time
    
    # We output statistics to stderr (ERP can read)
    sys.stderr.write(f"\n--- STATISTICS ---\n")
    sys.stderr.write(f"Execution time: {elapsed:.3f} sec\n")
    sys.stderr.write(f"LLM calls: {stats.llm_call_count}\n")
    sys.stderr.write(f"LLM tokens: {stats.llm_total_tokens}\n")
    sys.stderr.write(f"API calls: {stats.api_call_count}\n")
    sys.stderr.write(f"API tokens: {stats.api_total_tokens}\n")
    sys.stderr.write(f"Number of positions: {len(json.loads(result_json)) if result_json.startswith('[') else 0}\n")
    sys.stderr.write(f"user: {user}\n")
    return result_json


if __name__ == "__main__":
    # Test run
    test_text = """
    Масло сливочное 82% - 3 шт
    Коліно зварне D355 - 2шт
    Кран кульовий DN50
    """
    
    result = parse_order_with_stats(test_text)
    print("\nResult:")
    print(result)
    print("\nBeautiful output:")
    print(json.dumps(json.loads(result), ensure_ascii=False, indent=2))