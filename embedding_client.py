# embedding_client.py

from datetime import datetime
import time
import requests
from typing import List, Dict, Any, Optional


class OpenAIEmbeddingClient:
    """
    A universal client for receiving embeddings
from the OpenAI-compatible API.

Each batch is executed a maximum of RETRY_COUNT times.
If an error or incomplete response occurs, the entire batch is repeated.
    """

    RETRY_COUNT = 10
    RETRY_INTERVAL = 10  # seconds

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        default_headers: Optional[Dict] = None
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.default_headers = default_headers or {}

        self.url = f"{self.base_url}/embeddings"

        self.reset_stats()

    def reset_stats(self):
        """Reset statistics"""

        self.total_requests = 0
        self.total_tokens = 0
        self.total_time = 0.0

        self.last_response = None

        self.request_history = []

        # All errors are saved here
        self.errors = []

    def embed(
        self,
        texts: List[str],
        batch_size: int = 128
    ) -> List[List[float]]:
        """
        Getting embeddings for a list of texts.

        Each batch must return exactly as many embeddings
        as there were texts.

        If after 10 attempts the batch is not received —
        an exception is thrown and the entire operation is terminated.
        """

        if not texts:
            return []

        all_embeddings = []

        total_batches = (
            len(texts) + batch_size - 1
        ) // batch_size

        for batch_number, i in enumerate(
            range(0, len(texts), batch_size),
            start=1
        ):

            batch = texts[i:i + batch_size]

            batch_embeddings = self._embed_batch(
                batch,
                batch_number=batch_number,
                total_batches=total_batches
            )

            # Critical review
            if len(batch_embeddings) != len(batch):

                raise RuntimeError(
                    f"Batch {batch_number}/{total_batches}: "
                    f"expected {len(batch)} embeddings, "
                    f"received {len(batch_embeddings)}"
                )

            all_embeddings.extend(batch_embeddings)

        # Final protection
        if len(all_embeddings) != len(texts):

            raise RuntimeError(
                f"Number of embeddings does not match "
                f"the number of texts: "
                f"expected {len(texts)}, "
                f"received {len(all_embeddings)}"
            )

        return all_embeddings

    def _embed_batch(
        self,
        texts: List[str],
        batch_number: int = 0,
        total_batches: int = 0
    ) -> List[List[float]]:
        """
       Sends a single batch to the API.

If any error occurs or the response is incomplete,
the request is retried up to RETRY_COUNT times.
        """

        payload = {
            "model": self.model,
            "input": texts
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        headers.update(self.default_headers)

        last_error = None

        for attempt in range(
            1,
            self.RETRY_COUNT + 1
        ):

            start_time = time.time()

            self.total_requests += 1

            try:

                response = requests.post(
                    self.url,
                    json=payload,
                    headers=headers,
                    timeout=60
                )

                request_time = time.time() - start_time

                self.total_time += request_time
                self.last_response = response

                # -------------------------------------------------
                # HTTP error
                # -------------------------------------------------

                if response.status_code != 200:

                    response_text = response.text[:2000]

                    error_info = {
                        "type": "HTTPError",
                        "batch": batch_number,
                        "total_batches": total_batches,
                        "attempt": attempt,
                        "max_attempts": self.RETRY_COUNT,
                        "status_code": response.status_code,
                        "message": response_text,
                        "time_seconds": round(
                            request_time,
                            3
                        )
                    }

                    self.errors.append(error_info)

                    last_error = (
                        f"HTTP {response.status_code}: "
                        f"{response_text}"
                    )

                else:

                    # -------------------------------------------------
                    # JSON parsing
                    # -------------------------------------------------

                    try:
                        result = response.json()

                    except Exception as e:

                        error_info = {
                            "type": "JSONDecodeError",
                            "batch": batch_number,
                            "total_batches": total_batches,
                            "attempt": attempt,
                            "max_attempts": self.RETRY_COUNT,
                            "message": str(e),
                            "response": response.text[:2000]
                        }

                        self.errors.append(error_info)

                        last_error = (
                            f"Failed to parse JSON: {e}"
                        )

                    else:

                        # -------------------------------------------------
                        # Getting embeddings
                        # -------------------------------------------------

                        embeddings = []

                        data_items = result.get(
                            "data",
                            []
                        )

                        if isinstance(
                            data_items,
                            list
                        ):

                            data_items.sort(
                                key=lambda x: x.get(
                                    "index",
                                    0
                                )
                            )

                            for item in data_items:

                                if "embedding" in item:
                                    embeddings.append(
                                        item["embedding"]
                                    )

                        # -------------------------------------------------
                        # Tokens statistics
                        # -------------------------------------------------

                        usage = result.get(
                            "usage",
                            {}
                        )

                        tokens = usage.get(
                            "total_tokens",
                            0
                        )

                        self.total_tokens += tokens

                        # -------------------------------------------------
                        # Checking the number of embeddings
                        # -------------------------------------------------

                        if len(embeddings) != len(texts):

                            error_info = {
                                "type": "IncompleteEmbeddings",
                                "batch": batch_number,
                                "total_batches": total_batches,
                                "attempt": attempt,
                                "max_attempts": self.RETRY_COUNT,
                                "expected": len(texts),
                                "received": len(embeddings),
                                "tokens": tokens,
                                "time_seconds": round(
                                    request_time,
                                    3
                                )
                            }

                            self.errors.append(
                                error_info
                            )

                            last_error = (
                                f"Received "
                                f"{len(embeddings)} "
                                f"embeddings instead of "
                                f"{len(texts)}"
                            )

                        else:

                            # -------------------------------------------------
                            # Successful request
                            # -------------------------------------------------

                            self.request_history.append({
                                "batch": batch_number,
                                "attempt": attempt,
                                "texts_count": len(texts),
                                "time": round(
                                    request_time,
                                    3
                                ),
                                "tokens": tokens,
                                "model": result.get(
                                    "model",
                                    self.model
                                )
                            })

                            return embeddings

            except requests.exceptions.RequestException as e:

                request_time = time.time() - start_time

                self.total_time += request_time

                error_info = {
                    "type": type(e).__name__,
                    "batch": batch_number,
                    "total_batches": total_batches,
                    "attempt": attempt,
                    "max_attempts": self.RETRY_COUNT,
                    "message": str(e),
                    "time_seconds": round(
                        request_time,
                        3
                    )
                }

                self.errors.append(
                    error_info
                )

                last_error = str(e)

            except Exception as e:

                request_time = time.time() - start_time

                self.total_time += request_time

                error_info = {
                    "type": type(e).__name__,
                    "batch": batch_number,
                    "total_batches": total_batches,
                    "attempt": attempt,
                    "max_attempts": self.RETRY_COUNT,
                    "message": str(e),
                    "time_seconds": round(
                        request_time,
                        3
                    )
                }

                self.errors.append(
                    error_info
                )

                last_error = str(e)

            # ---------------------------------------------------------
            # If the attempt fails
            # ---------------------------------------------------------

            if attempt < self.RETRY_COUNT:

                time.sleep(
                    self.RETRY_INTERVAL
                )

        # -------------------------------------------------------------
        # 10 attempts ended in failure
        # -------------------------------------------------------------

        raise RuntimeError(
            f"Batch {batch_number}/{total_batches} "
            f"failed to retrieve after "
            f"{self.RETRY_COUNT} attempts. "
            f"Last error: {last_error}"
        )

    def embed_single(
        self,
        text: str
    ) -> Optional[List[float]]:

        result = self.embed(
            [text]
        )

        return result[0] if result else None

    def get_stats(
        self
    ) -> Dict[str, Any]:

        avg_time = (
            self.total_time /
            self.total_requests
            if self.total_requests > 0
            else 0
        )

        avg_tokens = (
            self.total_tokens /
            self.total_requests
            if self.total_requests > 0
            else 0
        )

        return {
            "provider": self.base_url,
            "model": self.model,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_time": f"{self.total_time:.3f} сек",
            "avg_time_per_request": (
                f"{avg_time:.3f} сек"
            ),
            "avg_tokens_per_request": (
                f"{avg_tokens:.1f}"
            ),
            "errors": self.errors,
            "request_history": self.request_history
        }