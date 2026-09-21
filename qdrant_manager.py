# qdrant_manager.py
"""
Client for working with Qdrant vector database.
Supports collection creation, adding points and searching.
"""

import time
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
    Payload
)

from config import QDRANT_CONFIG


class QdrantManager:
    """Client for working with Qdrant"""
    
    def __init__(self, host: str = "localhost", port: int = 6333):
        """
        Initialize the Qdrant client
        
        Arguments:
            host: host of the Qdrant server
            port: port of the Qdrant server
        """
        self.client = QdrantClient(host=host, port=port)
        self.host = host
        self.port = port

        self.errors = []
        self.upsert_history = []
    
    def collection_exists(self, collection_name: str) -> bool:
        """Checks if a collection exists"""
        try:
            self.client.get_collection(collection_name)
            return True
        except Exception:
            return False
    
    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "Cosine"
    ):
        """
        Creates a collection in Qdrant.

Arguments:
collection_name: collection name
vector_size: vector size
distance: distance metric (Cosine, Dot, Euclid)
        """
        # Delete the existing collection if it exists.
        if self.collection_exists(collection_name):
            #print(f"  ℹCollection '{collection_name}' already exists, delete it...")
            self.client.delete_collection(collection_name)
        
        # We are creating a new collection
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        #print(f"✓ Collection '{collection_name}' created (dimensions: {vector_size})")

        
    def add_points(
        self,
        collection_name: str,
        points: List[Dict[str, Any]],
        batch_size: int = 100,
        retry_count: int = 10,
        retry_interval: int = 20
    ):
        """
        Adds points to Qdrant in batches.

        Each batch is retried up to retry_count times if an error occurs.

        Resubmitting the batch is safe,
        since upsert is used with the same IDs.
        """

        total = len(points)

        if total == 0:
            return

        total_batches = (
            total + batch_size - 1
        ) // batch_size

        for batch_number, i in enumerate(
            range(0, total, batch_size),
            start=1
        ):

            batch = points[i:i + batch_size]

            points_struct = [
                PointStruct(
                    id=point['id'],
                    vector=point['vector'],
                    payload=point['payload']
                )
                for point in batch
            ]

            batch_start_time = time.time()

            last_error = None
            successful_attempt = None

            # ---------------------------------------------------------
            # Attempts to download batch
            # ---------------------------------------------------------

            for attempt in range(
                1,
                retry_count + 1
            ):

                attempt_start = time.time()

                try:

                    self.client.upsert(
                        collection_name=collection_name,
                        points=points_struct
                    )

                    attempt_time = (
                        time.time() - attempt_start
                    )

                    successful_attempt = attempt

                    self.upsert_history.append({
                        "batch": batch_number,
                        "total_batches": total_batches,
                        "points": len(batch),
                        "first_point": i + 1,
                        "last_point": min(
                            i + batch_size,
                            total
                        ),
                        "attempt": attempt,
                        "time_seconds": round(
                            attempt_time,
                            3
                        ),
                        "status": "success"
                    })

                    last_error = None

                    break

                except Exception as e:

                    attempt_time = (
                        time.time() - attempt_start
                    )

                    last_error = e

                    self.errors.append({
                        "type": type(e).__name__,
                        "batch": batch_number,
                        "total_batches": total_batches,
                        "attempt": attempt,
                        "max_attempts": retry_count,
                        "points": len(batch),
                        "first_point": i + 1,
                        "last_point": min(
                            i + batch_size,
                            total
                        ),
                        "message": str(e),
                        "time_seconds": round(
                            attempt_time,
                            3
                        )
                    })

                    if attempt < retry_count:

                        time.sleep(
                            retry_interval
                        )

            # ---------------------------------------------------------
            # Batch failed to load
            # ---------------------------------------------------------

            if last_error is not None:

                raise RuntimeError(
                    f"Loading error in Qdrant: "
                    f"batch {batch_number}/{total_batches}, "
                    f"points {i + 1}-"
                    f"{min(i + batch_size, total)}, "
                    f"after {retry_count} attempts. "
                    f"Last error: "
                    f"{type(last_error).__name__}: "
                    f"{last_error}"
                )

        # All batches have been successfully downloaded.
    
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs a vector search.

        Arguments:
        collection_name: collection name
        query_vector: query vector
        limit: number of results
        filter_conditions: payload filter (e.g., {"mark": "Walraven"})
        score_threshold: confidence threshold (0.0 - 1.0)

        Returns:
        list of results with metadata
        """
        # Building a filter
        query_filter = None
        if filter_conditions:
            conditions = []
            for key, value in filter_conditions.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            query_filter = Filter(must=conditions)
        
        # Performing the search
        response = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold
      )
        results = response.points
        
        # Formatting the results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.id,
                "score": result.score,
                "payload": result.payload,
                "confidence": result.score,  # For compatibility with ChromaDB
                "confidence_percent": f"{result.score * 100:.1f}%"
            })
        
        return formatted_results
    
    def delete_collection(self, collection_name: str):
        """Deletes a collection"""
        if self.collection_exists(collection_name):
            self.client.delete_collection(collection_name)
            #print(f"✓ Collection '{collection_name}' deleted")
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Returns information about the collection"""
        if not self.collection_exists(collection_name):
            return {"exists": False}
        
        info = self.client.get_collection(collection_name)
        return {
            "exists": True,
            "name": collection_name,
            "points_count": info.points_count,
            "vectors_count": getattr(info, 'vectors_count', None) or info.points_count,
            "status": getattr(info, 'status', None),
        }