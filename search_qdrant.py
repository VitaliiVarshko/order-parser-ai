# search_qdrant.py
"""
Search for products through Qdrant with hybrid ranking and full statistics.
"""

import re
import time
import json
import os
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from typing import Dict, Any, List

from config import (
    
    EMBEDDING_PROVIDER,
    EMBEDDING_CONFIG,
    QDRANT_CONFIG,
    CONFIDENCE_THRESHOLD,
    SEARCH_RESULTS,
    LLM_PROVIDER,
    LLM_CONFIG,
    LOG_DIR, 
    LOG_ENABLED
)
from embedding_client import OpenAIEmbeddingClient
from qdrant_manager import QdrantManager

collection_name = QDRANT_CONFIG["collection_name"]

# --- Search statistics ---
class SearchStats:

    def __init__(self):
        self.total_items = 0
        self.total_queries = 0
        self.total_time = 0.0

        self.last_error = None
        # statistics API
        self.api_call_count = 0
        self.api_total_tokens = 0
        self.api_total_time = 0.0
        self.api_model = ""
        self.api_provider = ""

        # LLM statistics
        self.llm_call_count = 0
        self.llm_total_tokens = 0
        self.llm_total_time = 0.0
    
    def add_llm_call(self, time_sec: float, tokens: int):
        self.llm_call_count += 1
        self.llm_total_time += time_sec
        self.llm_total_tokens += tokens


    
    def add_api_call(self, time_sec: float, tokens: int, model: str, provider: str):
        self.api_call_count += 1
        self.api_total_time += time_sec
        self.api_total_tokens += tokens
        self.api_model = model
        self.api_provider = provider
    
    def add_query_time(self, time_sec: float):
        self.total_queries += 1
        self.total_time += time_sec
    
    def print_stats(self):
        print("\n" + "=" * 60)
        print("📊 EXECUTION STATISTICS")
        print("=" * 60)
        print(f"  Total items: {self.total_items}")
        print(f"  Total queries: {self.total_queries}")
        print(f"  Total time: {self.total_time:.2f} sec")
        if self.total_queries > 0:
            print(f"  Average time per query: {self.total_time / self.total_queries:.2f} sec")
        
        print("\n--- API Embedding Statistics ---")
        print(f"  Provider: {self.api_provider}")
        print(f"  Model: {self.api_model}")
        print(f"  Total calls: {self.api_call_count}")
        print(f"  Total tokens: {self.api_total_tokens}")
        print(f"  Total time: {self.api_total_time:.3f} sec")
        if self.api_call_count > 0:
            print(f"  Average time per call: {self.api_total_time / self.api_call_count:.3f} sec")
            print(f"  Average tokens per call: {self.api_total_tokens / self.api_call_count:.1f}")

        print("\n--- LLM Statistics (Order Parsing) ---")
        print(f"  Total calls: {self.llm_call_count}")
        print(f"  Total tokens: {self.llm_total_tokens}")
        print(f"  Total time: {self.llm_total_time:.3f} sec")
        if self.llm_call_count > 0:
            print(f"  Average time per call: {self.llm_total_time / self.llm_call_count:.3f} sec")
            print(f"  Average tokens per call: {self.llm_total_tokens / self.llm_call_count:.1f}")


stats = SearchStats()

# --- Auxiliary functions ---
def extract_numbers(text: str) -> Set[str]:
    """Extracts all numbers from the text"""
    return set(re.findall(r'\d+', text))


def extract_keywords(text: str) -> Set[str]:
    """Extracts keywords from the text"""
    tech_patterns = []
    
    tech_patterns.extend(re.findall(r'\d+/\d+["\']?', text))
    tech_patterns.extend(re.findall(r'\d+[хx]\d+["\']?', text))
    tech_patterns.extend(re.findall(r'\d+["\']', text))
    tech_patterns.extend(re.findall(r'\b[A-Z]{2,}\d+\b', text))
    tech_patterns.extend(re.findall(r'\b[A-Za-zА-Яа-я]+[-]?\d+[-]?[A-Za-zА-Яа-я]*\b', text))
    tech_patterns.extend(re.findall(r'[Øø]\s*\d+[\.,]?\d*', text))
    
    words = re.findall(r'[а-щА-ЩЬьЮюЯяІіЇїЄєҐґa-zA-Z]+', text)
    
    result = set()
    
    for item in tech_patterns:
        item = item.lower().strip()
        if len(item) >= 2:
            result.add(item)
    
    for word in words:
        word = word.lower()
        if len(word) < 2:
            continue
        if word.isdigit():
            continue
        result.add(word)
    
    return result


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


# --- Search products ---
def search_products(
    qdrant: QdrantManager,
    embedding_client: OpenAIEmbeddingClient,
    query_text: str,
    mark_filter: Optional[str] = None,
    limit: int = SEARCH_RESULTS,
    debug: bool = True
) -> List[Dict[str, Any]]:
    """
    Searches for products through Qdrant with hybrid ranking
    
    Algorithm:
    1. Get the query embedding via API (collect statistics)
    2. Search in Qdrant by vector (with brand filter)
    3. Get all products (or many) for additional ranking
    4. Rank by matching numbers and keywords
    5. Return top results
    """
    global stats

    query_text1 = query_text.lower().strip()
    
    # --- Step 1: Get the query embedding ---
    start_time = time.time()
    query_embedding = embedding_client.embed_single(query_text1)
    elapsed = time.time() - start_time
    
    if not query_embedding:
        print(f"  ❌ Unable to get embedding for request")
        return []
    
    # We save statistics
    api_stats = embedding_client.get_stats()
    stats.add_api_call(
        elapsed,
        api_stats.get('total_tokens', 0),
        api_stats.get('model', 'unknown'),
        api_stats.get('provider', 'unknown')
    )
    
    if debug:
        print(f"  📊 API: {elapsed:.3f}с, tokens: {api_stats.get('total_tokens', 0)}")
    
    # --- Step 2: Search in Qdrant ---
    search_start = time.time()
    
    # First, we search for the top products to collect semantic evaluation
    filter_conditions = None
    if mark_filter and mark_filter.strip():
        filter_conditions = {"mark": mark_filter.strip()}
    else:
        # If the brand is not specified — we only search for products with an empty brand
        filter_conditions = {"mark": ""}

    # Getting more results for ranking
    qdrant_results = qdrant.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=40000,#limit * 5,  # We take more for subsequent ranking
        filter_conditions=filter_conditions
    )
    
    search_time = time.time() - search_start
    stats.add_query_time(search_time)
    
    if debug and not qdrant_results:
        print(f"  ❌ No results found")
        return []
    
    if debug:
        print(f"  🎯 Results found: {len(qdrant_results)}")
    
    # --- Step 3: Ranking ---
    query_numbers = extract_numbers(query_text1)
    query_keywords = extract_keywords(query_text1)
    
    if debug:
        print(f"  🔢 Numbers in query: {query_numbers} (total: {len(query_numbers)})")
        print(f"  📝 Keywords: {query_keywords}")
    
    total_query_numbers = len(query_numbers)
    total_query_keywords = len(query_keywords)
    
    # Rank the results
    ranked_results = []
    
    for result in qdrant_results:
        name = result['payload'].get('name', '')
        name_lower = name.lower()
        
        item_numbers = extract_numbers(name_lower)
        item_keywords = extract_keywords(name_lower)
        
        common_numbers = query_numbers & item_numbers
        common_keywords = query_keywords & item_keywords
        
        number_match_count = len(common_numbers)
        keyword_match_count = len(common_keywords)

        it_koef=0
        koef_keywords=0.40
        koef_numbers=0.40
        koef_semantic=0.20

        
        # Bonus for keywords (up to 50%)
        if total_query_keywords > 0:
            keyword_ratio = keyword_match_count / total_query_keywords
            keyword_bonus = keyword_ratio * koef_keywords
            it_koef=it_koef+koef_keywords
        else:
            keyword_bonus = 0.0
        
        # Bonus for numbers (up to 40%)
        if total_query_numbers > 0:
            number_ratio = number_match_count / total_query_numbers
            number_bonus = number_ratio * koef_numbers
            it_koef=it_koef+koef_numbers
        else:
            number_bonus = 0.0
        
        # Semantic bonus (up to 10%)
        semantic_score = result['score']  # This is already the cosine distance from Qdrant
        semantic_bonus = semantic_score * koef_semantic

        it_koef=it_koef+koef_semantic #total =1
        
        # Final confidence
        final_confidence = min((keyword_bonus + number_bonus + semantic_bonus) / it_koef, 1.0)

        final_confidence = max(final_confidence, 0.0)  # Ensure it's not less than 0
        
        ranked_results.append({
            "id": result['id'],
            "code": result['payload'].get('code', ''),
            "name": name,
            "mark": result['payload'].get('mark', ''),
            "semantic_score": semantic_score,
            "semantic_bonus": semantic_bonus,
            "keyword_bonus": keyword_bonus,
            "keyword_match_count": keyword_match_count,
            "number_bonus": number_bonus,
            "number_match_count": number_match_count,
            "confidence": final_confidence,
            "confidence_percent": f"{final_confidence * 100:.1f}%",
            "common_numbers": list(common_numbers) if common_numbers else [],
            "common_keywords": list(common_keywords) if common_keywords else [],
            "total_query_keywords": total_query_keywords,
            "total_query_numbers": total_query_numbers
        })
    
    # Sort by final confidence
    ranked_results.sort(key=lambda x: x['confidence'], reverse=True)
    
    if debug:
        print(f"\n  📊 Top 3 products by final confidence:")
        for i, item in enumerate(ranked_results[:3], 1):
            print(f"    {i}. Code: {item['code']} | Confidence: {item['confidence_percent']}")
            print(f"       Keywords: {item['keyword_match_count']}/{item['total_query_keywords']} | Numbers: {item['number_match_count']}/{item['total_query_numbers']}")
            print(f"       Name: {item['name'][:70]}")
    
    return ranked_results[:limit]


# --- Parsing quantity ---
def extract_quantity_and_name(line: str) -> tuple:
    line = line.strip()
    if not line:
        return "", "1", ""
    
    pattern_decimal = r'\s+-\s+(\d+)\s*([а-яА-Яa-zA-Z\.\/\-]*)?\.?\s*$'
    match = re.search(pattern_decimal, line)
    
    if match:
        qty = match.group(1)
        unit = match.group(2) if match.group(2) else ""
        name = re.sub(r'\s+-\s+\d+\s*[а-яА-Яa-zA-Z\.\/\-]*\.?\s*$', '', line)
        name = name.strip()
        return name, qty, unit
    
    pattern_end = r'\s+(\d+)\s*([а-яА-Яa-zA-Z\.\/\-]*)?\.?\s*$'
    match = re.search(pattern_end, line)
    
    if match:
        qty = match.group(1)
        unit = match.group(2) if match.group(2) else ""
        before_number = line[:match.start()].strip()
        if before_number and (before_number.endswith('х') or before_number.endswith(',') or before_number.endswith('x')):
            pass
        else:
            name = re.sub(r'\s+\d+\s*[а-яА-Яa-zA-Z\.\/\-]*\.?\s*$', '', line)
            name = name.strip()
            if name:
                return name, qty, unit
    
    return line, "1", ""



def parse_items(text: str) -> List[Dict[str, str]]:
    """
    Parses multi-line order text via the LLM API.
    """
    global stats
    
    import requests
    import json
    
    # Get the LLM config
    llm_config = LLM_CONFIG.get(LLM_PROVIDER, {})
    if not llm_config.get('api_key'):
        stats.last_error ="❌ Error: API key for LLM not found"
        return []
    
    url = f"{llm_config['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {llm_config['api_key']}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""
Ты — помощник менеджера по закупкам. 
Заказ клиента - список товаров с количеством. 
Количество идет в конце строки после дефиса или пробела, с единицами измерения или без. 
Указание названия изделия может быть отдельно в строке (1 или 2 слова) без количества и без размеров, это служебная строка, не надо ее брать как товар с кол-вом, она только указывает название для нижестоящих строк. 
Ниже идут строки с этим изделием, но без названия (которое уже задано выше), а только с размерами (числа, спец символы и одиночные буквы). 
Если в строке есть полноценные слова (не менее трех букв подряд), это уже своё название, название из выше стоящих слов подставлять не надо.
Если товар указан полностью в какой-то строке, то в строках ниже могут быть только отличающиеся размеры, без указания изделия.
Если строка содержит полноценное слово (не менее 3 букв подряд), то это начало нового товара. Не использовать название из предыдущих строк.
Если строка состоит только из размера/диаметра/обозначения (50, 50х90, 1 1/4", 20х45, 3/4" и т.п.) и количества, то подставлять последнее запомненное название товара.
Если строка содержит только название (1–2 слова) без количества, запомнить его и использовать для следующих строк.

Твоя задача — извлечь список товаров из текста заказа.

Правила:
1. Каждая позиция — отдельная строка, кроме случаев, когда это отдельная строка содержит только название для нижестоящих строк.
2. Для каждой позиции укажи название товара и количество, кроме случаев, когда это отдельная строка содержит только название для нижестоящих строк.
3. Если количество не указано явно — поставь 1, кроме случаев, когда это отдельная строка содержит только название для нижестоящих строк.
4. Сохрани оригинальное написание названий.
5. Исходную строку оставляй в поле "original".
6. Ответ должен быть ТОЛЬКО в формате JSON: массив объектов с полями "original", "name" и "qty".

Пример входа:
"
Коліно 
50х90 - 100 шт
50х45 - 50 шт
Кліпса 50 - 80шт
40 - 120
Трійник 50х32х50 20 комп.
Рідкий фум геб
"
В Примере входа строка "Коліно" — это отдельная служебная строка, которая задаёт название для нижестоящих строк с размерами. НЕ надо брать его как отдельную строку для выходного массива.

Пример выхода:
[
  {{"original": "Коліно 50х90 - 100 шт", "name": "Коліно 50х90", "qty": "100"}},
  {{"original": "Коліно 50х45 - 50 шт", "name": "Коліно 50х45", "qty": "50"}},
  {{"original": "Кліпса 50 - 80шт", "name": "Кліпса 50", "qty": "80"}},
  {{"original": "Кліпса 40 - 120", "name": "Кліпса 40", "qty": "120"}},
  {{"original": "Трійник 50х32х50 20 комп.", "name": "Трійник 50х32х50", "qty": "20"}},
  {{"original": "Рідкий фум геб", "name": "Рідкий фум геб", "qty": "1"}}
]

Текст заказа:
{text}

Ответ ТОЛЬКО в формате JSON, без пояснений.
"""
    
    payload = {
        "model": llm_config.get('model', 'mistral-small-latest'),
        "messages": [
            {"role": "system", "content": "Ты — помощник по извлечению товаров из заказов. Отвечай только JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 2000
    }
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        elapsed = time.time() - start_time
        
        if response.status_code != 200:
            print(f"❌ Error LLM: {response.status_code}")
            print(f"   {response.text}")

            stats.last_error = f"❌ Error LLM: {response.status_code} - {response.text}"
            return []
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        # Statistics
        usage = result.get('usage', {})
        tokens = usage.get('total_tokens', 0)
        stats.add_llm_call(elapsed, tokens)
        print(f"  📊 LLM: {elapsed:.2f}с, tokens: {tokens}")
        
        # Parse JSON
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        items = json.loads(content)
        
        parsed = []
        for item in items:
            parsed.append({
                "name": item.get('name', '').strip(),
                "qty": str(item.get('qty', '1')),
                "unit": "",
                "original": item.get('original', '').strip()
            })
        
        return parsed
        
    except Exception as e:
        stats.last_error =f"❌ Error parsing via LLM: {e}"
        return []




# --- Table formatting ---
def print_results_table(results: List[Dict]):
    print("\n" + "=" * 100)
    print("📊 FINAL TABLE OF RESULTS")
    print("=" * 100)
    
    headers = ["№", "Запрос", "Код товара", "Уверенность", "Кол-во", "Марка", "Статус"]
    widths = [4, 40, 14, 12, 8, 18, 20]
    
    header_line = ""
    for i, h in enumerate(headers):
        header_line += f"{h:<{widths[i]}} "
    print(header_line)
    print("-" * 100)
    
    for idx, res in enumerate(results, 1):
        if res['best_match']:
            code = res['best_match']['code']
            confidence = res['best_match']['confidence_percent']
            mark = res['best_match']['mark']
            if res['best_match']['confidence'] >= CONFIDENCE_THRESHOLD:
                status = "✅ НАЙДЕН"
            else:
                status = "⚠️ НИЗКАЯ УВЕРЕННОСТЬ"
        else:
            code = "НЕ НАЙДЕН"
            confidence = "-"
            mark = "-"
            status = "❌ НЕ НАЙДЕН"
        
        query_text = res['name'][:37]
        if len(res['name']) > 37:
            query_text += "..."
        
        row = f"{idx:<{widths[0]}} {query_text:<{widths[1]}} {code:<{widths[2]}} {confidence:<{widths[3]}} {res['qty']:<{widths[4]}} {mark:<{widths[5]}} {status:<{widths[6]}}"
        print(row)
    
    print("=" * 100)
    print(f"\n📌 Status Legend:")
    print(f"  ✅ FOUND - confidence >= {CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"  ⚠️ LOW CONFIDENCE - confidence < {CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"  ❌ NOT FOUND - no matches")




# --- function to call from ERP ---
def search_order_with_stats(items: List[Dict[str, str]], user: str = "") -> str:
    """
    Function to call from ERP via HTTP.
    Accepts a list of products from ERP, performs search.
    Returns a JSON string with results.
    """
    global stats
    
    start_time = time.time()
    start_dt = datetime.now().isoformat()
    
    # --- Structure for the log ---
    log_data = {
        "timestamp_start": start_dt,
        "user": user or "unknown",
        
        "input_items_count": len(items),
        "input_items": items,
        "results": [],
        "llm": {},
        "embedding": {},
        "errors": [],
        "timestamp_end": None,
        "duration_seconds": None
    }
    
    try:
        # We reset the statistics for the sake of measurement purity
        stats.api_call_count = 0
        stats.api_total_tokens = 0
        stats.api_total_time = 0.0
        stats.total_queries = 0
        stats.total_time = 0.0
        
        # --- Connecting to Qdrant ---
        qdrant = QdrantManager(
            host=QDRANT_CONFIG["host"],
            port=QDRANT_CONFIG["port"]
        )
        
        # Checking the collection
       
        info = qdrant.get_collection_info(collection_name)
        if not info['exists']:
            raise Exception(f"Collection '{collection_name}' not found")
        
        embedding_client = create_embedding_client()
        
        # --- Search Results ---
        results = []
        
        for item in items:
            query_text = item.get('name', '')
            mark = item.get('mark', '')
            qty = item.get('qty', '1')
            
            if not query_text:
                results.append({
                    "mark": mark,
                    "name": query_text,
                    "qty": qty,
                    "best_match": None,
                    "alternatives": [],
                    "error": "Пустое название товара"
                })
                continue
            
            # We are looking for products
            found = search_products(
                qdrant=qdrant,
                embedding_client=embedding_client,
                query_text=query_text,
                mark_filter=mark if mark else None,
                limit=SEARCH_RESULTS,
                debug=False  # Disabling console output
            )
            
            if found:
                best = found[0]
                results.append({
                    
                    "name": query_text,
                    "qty": qty,
                    "best_match": {
                        "code": best.get('code', ''),
                        "name": best.get('name', ''),
                        "mark": best.get('mark', ''),
                        "confidence": best.get('confidence_percent', '0%')
                    },
                    "alternatives": [
                        {
                            "code": alt.get('code', ''),
                            "name": alt.get('name', ''),
                            "mark": alt.get('mark', ''),
                            "confidence": alt.get('confidence_percent', '0%')
                        }
                        for alt in found[1:3]  # Only 2 alternatives
                    ]
                })
            else:
                results.append({
                    
                    "name": query_text,
                    "qty": qty,
                    "best_match": None,
                    "alternatives": []
                })
        
        # --- Saving Statistics ---
        log_data["embedding"] = {
            "calls": stats.api_call_count,
            "tokens": stats.api_total_tokens,
            "time_seconds": round(stats.api_total_time, 3),
            "model": stats.api_model or "unknown"
        } if stats.api_call_count > 0 else {}
        
        log_data["results"] = results
        log_data["timestamp_end"] = datetime.now().isoformat()
        log_data["duration_seconds"] = round(time.time() - start_time, 3)
        log_data["found_count"] = sum(1 for r in results if r.get('best_match'))
        
        # --- Saving Log ---
        save_search_log(log_data)
        
        # --- Returning Result ---
        return json.dumps(results, ensure_ascii=False)
        
    except Exception as e:
        error_msg = str(e)
        log_data["errors"].append({
            "type": type(e).__name__,
            "message": error_msg,
            "timestamp": datetime.now().isoformat()
        })
        log_data["timestamp_end"] = datetime.now().isoformat()
        log_data["duration_seconds"] = round(time.time() - start_time, 3)
        
        save_search_log(log_data)
        
        error_result = {"error": error_msg}
        return json.dumps(error_result, ensure_ascii=False)






def save_search_log(log_data: Dict[str, Any]):
    """Saves the search log to a JSON file."""
    if not LOG_ENABLED:
        return
    
    if not os.path.exists(LOG_DIR):
        try:
            os.makedirs(LOG_DIR)
        except:
            return
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    user = log_data.get('user', 'unknown')
    filename = f"search_{timestamp}_{user}.json"
    filepath = os.path.join(LOG_DIR, filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2, default=str)
    except:
        pass





# --- Main function ---
def process_order(text: str, mark: Optional[str] = None):
    global stats
    stats = SearchStats()
    
    print("=" * 60)
    print("🔍 PRODUCT SEARCH (QDRANT)")
    print("=" * 60)
    
    # 1. We create clients
    embedding_client = create_embedding_client()
    qdrant = QdrantManager(
        host=QDRANT_CONFIG["host"],
        port=QDRANT_CONFIG["port"]
    )
    
    print(f"ℹ Using provider: {EMBEDDING_PROVIDER}")
    print(f"  Model: {embedding_client.model}")
    print(f"  Qdrant: http://{QDRANT_CONFIG['host']}:{QDRANT_CONFIG['port']}")
    
    # 2. We check the collection
    
    info = qdrant.get_collection_info(collection_name)
    if not info['exists']:
        print(f"❌ Error: Collection '{collection_name}' not found")
        print(f"  First, run load_to_qdrant.py to load the data")
        return
    
    print(f"  Collection '{collection_name}': {info['points_count']} points")
    
    # 3. We parse the order
    items = parse_items(text)
    print(f"\n📋 Parsed items: {len(items)}")
    stats.total_items = len(items)
    
    if not items:
        print("❌ No products to search")
        return
    
    results = []
    
    for idx, item in enumerate(items, 1):
        print(f"\n--- Position {idx} ---")
        print(f"  Original: {item['original']}")
        print(f"  Name: {item['name']}")
        print(f"  Quantity: {item['qty']} {item['unit']}".strip())
        
        if mark:
            print(f"  Filter by brand: '{mark}'")
        
        found = search_products(
            qdrant=qdrant,
            embedding_client=embedding_client,
            query_text=item['name'],
            mark_filter=mark,
            limit=SEARCH_RESULTS
        )
        
        if found:
            best = found[0]
            status = "✅" if best['confidence'] >= CONFIDENCE_THRESHOLD else "⚠️"
            
            print(f"  {status} Best result:")
            print(f"    Code: {best['code']}")
            print(f"    Name: {best['name']}")
            print(f"    Brand: {best['mark']}")
            print(f"    confidence: {best['confidence_percent']}")
            print(f"    Matched numbers: {best.get('number_match_count', 0)}/{best.get('total_query_numbers', 0)}")
            print(f"    Matched keywords: {best.get('keyword_match_count', 0)}/{best.get('total_query_keywords', 0)}")
            
            if len(found) > 1:
                print(f"  Alternatives:")
                for alt in found[1:3]:
                    print(f"    - {alt['name']} ({alt['confidence_percent']})")
            
            results.append({
                "original": item['original'],
                "name": item['name'],
                "qty": f"{item['qty']} {item['unit']}".strip(),
                "best_match": best,
                "alternatives": found[1:] if len(found) > 1 else []
            })
        else:
            print(f"  ❌ No matches found")
            results.append({
                "original": item['original'],
                "name": item['name'],
                "qty": f"{item['qty']} {item['unit']}".strip(),
                "best_match": None,
                "alternatives": []
            })
    
    print_results_table(results)
    stats.print_stats()
    
    return results


# ============================================================
# 📝 TEST DATA
# ============================================================

if __name__ == "__main__":
    ORDER_TEXT = """
    Хомут 3" - 50шт
    2" - 50
    Бур 6 20наб.
    Анкер 
    16x45 - 10 компл.
    10х34 
    Труба скло 50 -80 м
Труба 40 - 120 м
Труба 32 - 240м
Труба 25 - 40м
Труба 20 - 140 м
Кліпса 50 - 80 шт
40 - 120 шт
32 - 240 шт
25 - 40 шт
20 - 140 шт
Хомут 1 1/2" - 40 шт
1 1/4" - 40 шт
1" - 40 шт
Рідкий фум геб - 10 шт
Пакля - 10 шт
Паста - 4 шт
    """
    
    MARK_FILTER = "Walraven"
    
    results = process_order(ORDER_TEXT, MARK_FILTER)