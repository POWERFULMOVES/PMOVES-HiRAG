import os
import logging
import numpy as np
import yaml
import cohere
import asyncio
from hirag import HiRAG, QueryParam
from dataclasses import dataclass
from hirag.base import BaseKVStorage
from hirag._utils import compute_args_hash
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration from YAML file
try:
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    logger.error("Error: config.yaml not found. Please create it with Cohere and HiRAG settings.")
    exit(1)
except yaml.YAMLError as e:
    logger.error(f"Error parsing config.yaml: {e}")
    exit(1)

# Extract Cohere configurations
try:
    COHERE_API_KEY = config['cohere']['api_key']
    COHERE_CHAT_MODEL = config['cohere']['model']
    COHERE_EMBEDDING_MODEL = config['cohere']['embedding_model']
    COHERE_EMBEDDING_DIM = config['cohere']['embedding_dim']
    # Optional: Use environment variables as fallback or override
    COHERE_API_KEY = os.environ.get("COHERE_API_KEY", COHERE_API_KEY)
except KeyError as e:
    logger.error(f"Missing key in config.yaml under 'cohere': {e}")
    exit(1)

if not COHERE_API_KEY:
    logger.error("Cohere API key not found in config.yaml or COHERE_API_KEY environment variable.")
    exit(1)

# Extract HiRAG configurations
try:
    HIRAG_WORKING_DIR = config['hirag']['working_dir']
    HIRAG_ENABLE_LLM_CACHE = config['hirag'].get('enable_llm_cache', True)
    HIRAG_ENABLE_HIERARCHICAL_MODE = config['hirag'].get('enable_hierachical_mode', True)
    HIRAG_EMBEDDING_BATCH_NUM = config['hirag'].get('embedding_batch_num', 16)
    HIRAG_EMBEDDING_FUNC_MAX_ASYNC = config['hirag'].get('embedding_func_max_async', 4)
    HIRAG_ENABLE_NAIVE_RAG = config['hirag'].get('enable_naive_rag', False)
    # Optional input file path from config
    INPUT_FILE_PATH = config.get('input_file', None)
except KeyError as e:
    logger.error(f"Missing key in config.yaml under 'hirag': {e}")
    exit(1)


# --- Embedding Function ---

@dataclass
class EmbeddingFunc:
    embedding_dim: int
    # Cohere doesn't explicitly publish a max token size for embed v3 like OpenAI does for its models.
    # We'll omit it here unless specific constraints are needed.
    # max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)

def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap an async function with attributes required by HiRAG."""
    def final_decorator(func) -> EmbeddingFunc:
        # Ensure the function is async
        if not asyncio.iscoroutinefunction(func):
            raise TypeError(f"The decorated function {func.__name__} must be async.")
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func
    return final_decorator

@wrap_embedding_func_with_attrs(embedding_dim=COHERE_EMBEDDING_DIM)
async def COHERE_embedding(texts: list[str]) -> np.ndarray:
    """Generates embeddings for a list of texts using Cohere API."""
    # Note: Cohere recommends using AsyncClient for concurrent requests
    co_async = cohere.AsyncClient(api_key=COHERE_API_KEY)
    try:
        # Determine input type based on typical HiRAG usage: 'search_document' for indexing.
        # HiRAG might call this for queries too; Cohere recommends 'search_query' for queries.
        # For simplicity here, we use 'search_document'. A more robust implementation
        # might inspect the call context or pass an input_type hint.
        response = await co_async.embed(
            texts=texts,
            model=COHERE_EMBEDDING_MODEL,
            input_type="search_document" # Use "search_query" when embedding single queries
        )
        # Ensure embeddings are numpy arrays
        embeddings = np.array(response.embeddings, dtype=np.float32)
        if embeddings.shape[0] != len(texts) or embeddings.shape[1] != COHERE_EMBEDDING_DIM:
            logger.error(f"Unexpected embedding shape: {embeddings.shape}. Expected ({len(texts)}, {COHERE_EMBEDDING_DIM})")
            # Handle error appropriately, maybe raise or return empty array
            raise ValueError("Embedding dimension mismatch or incorrect number of embeddings returned.")
        return embeddings
    except cohere.CohereError as e:
        logger.error(f"Cohere API error during embedding: {e}")
        # Re-raise or handle as needed; returning empty array might cause issues downstream
        raise
    except Exception as e:
        logger.error(f"Unexpected error during embedding: {e}")
        raise
    finally:
        # Ensure the async client session is closed
        await co_async.close()


# --- Model (Chat) Function ---

def _format_history_for_cohere(history_messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Converts OpenAI-style history to Cohere format."""
    cohere_history = []
    for msg in history_messages:
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        if role == "user":
            cohere_history.append({"role": "USER", "message": content})
        elif role == "assistant" or role == "model": # HiRAG might use 'model'
            cohere_history.append({"role": "CHATBOT", "message": content})
        # Silently ignore system messages here, handled by 'preamble' in co.chat
    return cohere_history

async def COHERE_model_if_cache(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict[str, str]] = [],
    **kwargs
) -> str:
    """Uses Cohere Chat API, checking cache first."""
    co_async = cohere.AsyncClient(api_key=COHERE_API_KEY)
    hashing_kv: Optional[BaseKVStorage] = kwargs.pop("hashing_kv", None)
    cache_key = None

    # Prepare request details for hashing and API call
    chat_history = _format_history_for_cohere(history_messages)
    # For hashing, combine relevant parts. Use a simplified representation.
    hash_payload = {
        "model": COHERE_CHAT_MODEL,
        "message": prompt,
        "chat_history": chat_history,
        "preamble": system_prompt,
        # Include other relevant kwargs if they affect the output significantly
        "temperature": kwargs.get("temperature", 0.3) # Example
    }

    # Check cache
    if hashing_kv is not None:
        cache_key = compute_args_hash(hash_payload)
        logger.debug(f"Checking cache for key: {cache_key}")
        cached_response = await hashing_kv.get_by_id(cache_key)
        if cached_response is not None and "return" in cached_response:
            logger.info(f"Cache hit for key: {cache_key}")
            await co_async.close() # Close client if returning from cache
            return cached_response["return"]
        else:
            logger.info(f"Cache miss for key: {cache_key}")


    # Call Cohere API
    try:
        logger.debug(f"Calling Cohere chat model: {COHERE_CHAT_MODEL}")
        response = await co_async.chat(
            model=COHERE_CHAT_MODEL,
            message=prompt,
            chat_history=chat_history,
            preamble=system_prompt,
            temperature=kwargs.get("temperature", 0.3), # Pass through relevant params
            # max_tokens=kwargs.get("max_tokens", None) # Example if needed
        )
        result_text = response.text

        # Store in cache if enabled
        if hashing_kv is not None and cache_key is not None:
             logger.debug(f"Storing response in cache for key: {cache_key}")
             await hashing_kv.upsert(
                 {cache_key: {"return": result_text, "model": COHERE_CHAT_MODEL}}
             )

        return result_text

    except cohere.CohereError as e:
        logger.error(f"Cohere API error during chat: {e}")
        raise # Re-raise to signal failure
    except Exception as e:
        logger.error(f"Unexpected error during chat: {e}")
        raise
    finally:
        # Ensure the async client session is closed
        await co_async.close()


# --- Main Execution Logic ---

async def main():
    """Initializes HiRAG with Cohere and performs indexing/querying."""

    logger.info("Initializing HiRAG with Cohere backend...")
    graph_func = HiRAG(working_dir=HIRAG_WORKING_DIR,
                      enable_llm_cache=HIRAG_ENABLE_LLM_CACHE,
                      embedding_func=COHERE_embedding,
                      best_model_func=COHERE_model_if_cache, # Use Cohere for both best and cheap
                      cheap_model_func=COHERE_model_if_cache,
                      enable_hierachical_mode=HIRAG_ENABLE_HIERARCHICAL_MODE,
                      embedding_batch_num=HIRAG_EMBEDDING_BATCH_NUM,
                      embedding_func_max_async=HIRAG_EMBEDDING_FUNC_MAX_ASYNC,
                      enable_naive_rag=HIRAG_ENABLE_NAIVE_RAG)

    # --- Indexing ---
    # Check if the working directory exists and might already be indexed.
    # HiRAG's insert might handle this, but explicit checks can be useful.
    if INPUT_FILE_PATH:
        if not os.path.exists(INPUT_FILE_PATH):
             logger.error(f"Input file not found: {INPUT_FILE_PATH}")
             return # Exit if input file specified but not found

        # Check if indexing might be needed (e.g., based on existence of index files)
        # For simplicity, we'll just run insert. Add more sophisticated checks if needed.
        logger.info(f"Indexing data from: {INPUT_FILE_PATH}")
        try:
            with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f:
                text_content = f.read()
            # Assuming insert is idempotent or handles re-indexing appropriately
            await graph_func.insert(text_content) # Use await for async insert if available
            logger.info("Indexing complete.")
        except Exception as e:
            logger.error(f"Error during indexing: {e}")
            return # Stop if indexing fails
    else:
        logger.warning("No input_file specified in config.yaml. Skipping indexing.")
        logger.warning("Ensure the working directory contains a pre-built index or run indexing manually.")


    # --- Querying ---
    query_text = "What are the main capabilities of this system?" # Example query
    logger.info(f"Performing HiRAG query: '{query_text}'")
    try:
        # Assuming query is async or HiRAG handles the async calls internally
        # If graph_func.query itself needs await: result = await graph_func.query(...)
        result = graph_func.query(query_text, param=QueryParam(mode="hi" if HIRAG_ENABLE_HIERARCHICAL_MODE else "naive"))
        logger.info("Query Result:")
        print(result) # Print the result directly
    except Exception as e:
        logger.error(f"Error during query: {e}")


if __name__ == "__main__":
    # Ensure event loop is running for async operations
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred in main execution: {e}")
