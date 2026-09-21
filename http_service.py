# http_service.py
import json
import multiprocessing
import sys

if sys.stdout:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if sys.stderr:
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI, Request
from pydantic import BaseModel
from order_parser import parse_order_with_stats
from search_qdrant import search_order_with_stats  
import uvicorn

app = FastAPI()

class OrderRequest(BaseModel):
    text: str
    user: str = ""  

class SearchRequest(BaseModel):
    items: list  # array of products from 1C
    user: str = ""

@app.post("/parse")
async def parse_order(request: OrderRequest):
    result = parse_order_with_stats(request.text, request.user)

    # Parse the result to check for errors
    try:
        data = json.loads(result)
        if isinstance(data, dict) and data.get("error"):
            # There is an error — return it explicitly
            return {"status": "error", "error": data.get("error"), "result": []}
    except:
        pass
    
    return {"status": "ok", "result": json.loads(result)}

    

@app.post("/search")
async def search_products(request: SearchRequest):
    """
    Receives input from 1C:
    {
        "items": [{"mark": "...", "name": "...", "qty": "..."}, ...],
        
        "user": "Иванов И.И."
    }
    Returns:
    [
        {
            "original": "...",
            "name": "...",
            "qty": "...",
            "best_match": {
                "code": "...",
                "name": "...",
                "mark": "...",
                "confidence": "..."
            },
            "alternatives": [...]
        },
        ...
    ]
    """
    result = search_order_with_stats(request.items, request.user)
    return json.loads(result)


@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    multiprocessing.freeze_support()  # ← for correct operation on Windows during assembly
    uvicorn.run(app, host="0.0.0.0", port=8000)