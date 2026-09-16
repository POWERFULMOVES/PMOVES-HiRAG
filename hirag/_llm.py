import numpy as np
import os
import cohere
from cohere import CohereAPIError, CohereConnectionError, CohereRateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from ._utils import compute_args_hash, wrap_embedding_func_with_attrs
from .base import BaseKVStorage
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

global_cohere_async_client = None


def get_cohere_async_client_instance():
    """Get or create an asynchronous Cohere client instance."""
    global global_cohere_async_client
    if global_cohere_async_client is None:
        api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            logger.warning("COHERE_API_KEY environment variable not set. Cohere calls will fail.")
            # Allow creation to proceed, but calls will likely fail, providing feedback.
            # Alternatively, raise an error: raise ValueError("COHERE_API_KEY not set")
        
        # Initialize the async client
        # Add timeout configurations if needed, e.g., timeout=60
        global_cohere_async_client = cohere.AsyncClient(
            api_key=api_key,
            # Consider adding client-side timeouts if appropriate
            # timeout=(10, 60) # (connect timeout, read timeout)
        )
        logger.info("Cohere AsyncClient initialized.")
    return global_cohere_async_client


def _format_chat_history_for_cohere(history_messages: list[dict]) -> list[dict]:
    """Converts a list of messages from OpenAI format to Cohere format."""
    cohere_history = []
    role_map = {"user": "USER", "assistant": "CHATBOT", "system": "SYSTEM"} # SYSTEM role may not map directly, handled by preamble
    for msg in history_messages:
        role = role_map.get(msg.get("role"))
        content = msg.get("content")
        if role and content and role != "SYSTEM": # System messages handled by preamble
             cohere_history.append({"role": role, "message": content})
        elif role == "SYSTEM":
            logger.warning("System messages in history are ignored; use the 'system_prompt' parameter instead for Cohere preamble.")
    return cohere_history


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((CohereRateLimitError, CohereConnectionError, CohereAPIError)),
    reraise=True # Reraise the exception after retries are exhausted
)
async def cohere_complete_if_cache(
    model: str | None = None,
    prompt: str | None = None,
    system_prompt: str | None = None,
    history_messages: list[dict] | None = None,
    **kwargs
) -> str:
    """
    Generates a completion using the Cohere API, with caching support.

    Args:
        model (str | None): The Cohere model ID (e.g., 'command-r'). Defaults to COHERE_CHAT_MODEL env var or 'command-r'.
        prompt (str | None): The user's prompt/message.
        system_prompt (str | None): The system prompt (preamble for Cohere).
        history_messages (list[dict] | None): A list of previous messages in OpenAI format [{'role': 'user'|'assistant', 'content': ...}].
        **kwargs: Additional arguments passed to the Cohere client's chat method (e.g., temperature, max_tokens)
                  and 'hashing_kv' for caching.

    Returns:
        str: The generated text content.

    Raises:
        CohereAPIError, CohereConnectionError, CohereRateLimitError: If API calls fail after retries.
        ValueError: If prompt is None.
    """
    if prompt is None:
        raise ValueError("Prompt cannot be None for cohere_complete_if_cache")

    cohere_async_client = get_cohere_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)

    # Determine model, preferring explicit > env var > default
    effective_model = model or os.environ.get("COHERE_CHAT_MODEL", "command-r")
    
    # Format history messages for Cohere API
    chat_history = _format_chat_history_for_cohere(history_messages or [])

    # Arguments for hashing and API call (excluding non-API kwargs like hashing_kv)
    api_args = {
        "model": effective_model,
        "message": prompt,
        "preamble": system_prompt,
        "chat_history": chat_history,
        **kwargs # Pass through other Cohere-specific args like temperature, max_tokens
    }
    
    # Filter out None values before hashing/calling API
    api_args_filtered = {k: v for k, v in api_args.items() if v is not None}


    if hashing_kv is not None:
        # Use filtered args for hashing to ensure consistency
        args_hash = compute_args_hash(api_args_filtered)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            logger.debug(f"Cache hit for Cohere completion (hash: {args_hash})")
            return if_cache_return["return"]
        logger.debug(f"Cache miss for Cohere completion (hash: {args_hash})")

    try:
        logger.debug(f"Calling Cohere chat API with args: {api_args_filtered}")
        response = await cohere_async_client.chat(**api_args_filtered)
        completion_text = response.text
        logger.debug(f"Received Cohere chat response. Text length: {len(completion_text)}")

    except (CohereAPIError, CohereConnectionError, CohereRateLimitError) as e:
        logger.error(f"Cohere API error during chat completion: {e}")
        raise # Reraise to trigger tenacity retry or final failure
    except Exception as e:
        logger.exception(f"An unexpected error occurred during Cohere chat completion: {e}")
        raise # Reraise unexpected errors

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": completion_text, "model": effective_model}}
        )
        # Assuming index_done_callback is for batching/finalizing writes
        await hashing_kv.index_done_callback()
        logger.debug(f"Cached Cohere completion result (hash: {args_hash})")

    return completion_text


async def cohere_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict] | None = None,
    model: str | None = None,
    **kwargs
) -> str:
    """
    High-level wrapper for Cohere chat completion using default settings.

    Args:
        prompt (str): The user's prompt/message.
        system_prompt (str | None): The system prompt (preamble for Cohere).
        history_messages (list[dict] | None): List of previous messages.
        model (str | None): Specific Cohere model to use. Overrides defaults.
        **kwargs: Additional arguments for cohere_complete_if_cache (including hashing_kv).

    Returns:
        str: The generated text content.
    """
    # Model resolution happens inside cohere_complete_if_cache
    return await cohere_complete_if_cache(
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

# --- Embedding Function ---

# Example dimensions for common Cohere v3 models
# See: https://docs.cohere.com/reference/embed
COHERE_EMBED_DIMS = {
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-light-v3.0": 384,
    "embed-english-v2.0": 4096,
    "embed-english-light-v2.0": 1024,
    "embed-multilingual-v2.0": 768,
}
# Recommended max tokens (not a hard limit enforced by API)
COHERE_EMBED_MAX_TOKENS = 512 # Cohere recommends under 512 for optimal quality

@wrap_embedding_func_with_attrs( # Decorator might need adjustment based on actual model used
    embedding_dim=COHERE_EMBED_DIMS.get(os.environ.get("COHERE_EMBEDDING_MODEL", "embed-english-v3.0"), 1024), # Default to common model dim
    max_token_size=COHERE_EMBED_MAX_TOKENS # Use recommended token size
)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((CohereRateLimitError, CohereConnectionError, CohereAPIError)),
    reraise=True
)
async def cohere_embedding(
    texts: list[str],
    model: str | None = None,
    input_type: str = "search_document",
    embedding_types: list[str] | None = None, # Allow overriding default ['float']
    hashing_kv: BaseKVStorage | None = None # Added for potential caching
) -> np.ndarray:
    """
    Generates embeddings for a list of texts using the Cohere API.

    Args:
        texts (list[str]): A list of strings to embed.
        model (str | None): The Cohere embedding model ID. Defaults to COHERE_EMBEDDING_MODEL env var or 'embed-english-v3.0'.
        input_type (str): Specifies the type of input passed to the model (v3+).
                          Examples: "search_document", "search_query", "classification", "clustering".
                          Defaults to "search_document".
        embedding_types (list[str] | None): Specifies the desired embedding types (e.g., ['float', 'int8']).
                                          Defaults to ['float'].
        hashing_kv (BaseKVStorage | None): Optional KV store for caching results.

    Returns:
        np.ndarray: A numpy array where each row is the embedding for the corresponding text.
                    Returns only the 'float' embeddings if multiple types are requested but caching is not implemented for multiple types.

    Raises:
        CohereAPIError, CohereConnectionError, CohereRateLimitError: If API calls fail after retries.
        ValueError: If texts list is empty.
    """
    if not texts:
        logger.warning("Received empty list of texts for embedding. Returning empty array.")
        return np.array([])
        # Alternatively: raise ValueError("Texts list cannot be empty for cohere_embedding")

    cohere_async_client = get_cohere_async_client_instance()

    # Determine model, preferring explicit > env var > default
    effective_model = model or os.environ.get("COHERE_EMBEDDING_MODEL", "embed-english-v3.0")
    
    # Default embedding types if not specified
    effective_embedding_types = embedding_types or ["float"]
    
    # Arguments for hashing and API call
    api_args = {
        "model": effective_model,
        "texts": texts,
        "input_type": input_type,
        "embedding_types": effective_embedding_types,
        # Add truncate parameter if needed, e.g., "truncate": "END"
    }

    # --- Caching Logic (Optional for Embeddings) ---
    args_hash = None
    if hashing_kv is not None:
        args_hash = compute_args_hash(api_args) # Hash includes all relevant params
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            logger.debug(f"Cache hit for Cohere embedding (hash: {args_hash})")
            # Assuming cache stores numpy array directly or can reconstruct it
            # This might need adjustment based on how caching_kv stores/retrieves complex types
            cached_data = if_cache_return.get("return")
            if isinstance(cached_data, list): # Simple check if it was stored as list
                 return np.array(cached_data)
            elif isinstance(cached_data, np.ndarray):
                 return cached_data
            else:
                 logger.warning(f"Cached embedding data format unexpected (hash: {args_hash}). Re-fetching.")
                 # Fall through to fetch if format is wrong
        else:
             logger.debug(f"Cache miss for Cohere embedding (hash: {args_hash})")
    # --- End Caching Logic ---

    try:
        logger.debug(f"Calling Cohere embed API with model '{effective_model}', {len(texts)} texts, input_type '{input_type}'")
        response = await cohere_async_client.embed(**api_args)
        
        # Extract embeddings - prioritizing 'float' if available
        if hasattr(response, 'embeddings') and 'float' in response.embeddings:
             embeddings_list = response.embeddings['float']
        elif hasattr(response, 'embeddings') and effective_embedding_types[0] in response.embeddings:
             # Fallback to the first requested type if float isn't there (e.g., if only 'int8' was requested)
             embeddings_list = response.embeddings[effective_embedding_types[0]]
             logger.warning(f"Returning '{effective_embedding_types[0]}' embeddings as 'float' was not found in response.")
        elif isinstance(response.embeddings, list): # Handle older API or potential variations
            embeddings_list = response.embeddings
            logger.warning("Cohere embed response format unexpected (expected dict with types), using direct list.")
        else:
            logger.error(f"Could not extract embeddings from Cohere response. Response keys: {list(response.embeddings.keys()) if hasattr(response, 'embeddings') and isinstance(response.embeddings, dict) else 'N/A'}")
            raise ValueError("Failed to extract embeddings from Cohere API response.")

        result_array = np.array(embeddings_list)
        logger.debug(f"Received Cohere embeddings. Shape: {result_array.shape}")

    except (CohereAPIError, CohereConnectionError, CohereRateLimitError) as e:
        logger.error(f"Cohere API error during embedding: {e}")
        raise
    except Exception as e:
        logger.exception(f"An unexpected error occurred during Cohere embedding: {e}")
        raise

    # --- Caching Save Logic ---
    if hashing_kv is not None and args_hash is not None:
         # Store as list for broader compatibility, can be adjusted
        await hashing_kv.upsert(
            {args_hash: {"return": result_array.tolist(), "model": effective_model}}
        )
        await hashing_kv.index_done_callback()
        logger.debug(f"Cached Cohere embedding result (hash: {args_hash})")
    # --- End Caching Save Logic ---
    
    # Dynamically update the decorator's attributes based on the actual model used, if needed
    # This part is complex as the decorator is applied at definition time.
    # A simpler approach is to ensure the decorator uses the default model's info,
    # or remove the dimension/token checks if they cause issues with dynamic models.
    # For now, we rely on the initial decorator values based on environment or defaults.
    # Optionally, log a warning if the used model's known dim differs from decorator's:
    # known_dim = COHERE_EMBED_DIMS.get(effective_model)
    # if known_dim and known_dim != cohere_embedding.embedding_dim:
    #     logger.warning(f"Model '{effective_model}' has dimension {known_dim}, but decorator uses {cohere_embedding.embedding_dim}.")

    return result_array

# Example Usage (can be removed or placed under if __name__ == "__main__":)
async def example_main():
    # Ensure COHERE_API_KEY is set as an environment variable
    if not os.environ.get("COHERE_API_KEY"):
        print("Please set the COHERE_API_KEY environment variable.")
        return

    print("--- Testing Cohere Completion ---")
    try:
        completion = await cohere_complete(
            prompt="What is the capital of France?",
            # system_prompt="Respond concisely.", # Optional Preamble
            # model="command-r-plus" # Optional: override default
        )
        print(f"Completion Result: {completion}")
    except Exception as e:
        print(f"Completion failed: {e}")

    print("--- Testing Cohere Embedding ---")
    try:
        texts_to_embed = ["hello world", "large language model"]
        embeddings = await cohere_embedding(
             texts=texts_to_embed,
             input_type="search_document", # Or "search_query", "classification", etc.
             # model="embed-english-v3.0" # Optional: override default
        )
        print(f"Embedding Result Shape: {embeddings.shape}")
        # print(f"First Embedding (first 5 dims): {embeddings[0][:5]}")
    except Exception as e:
        print(f"Embedding failed: {e}")

# if __name__ == "__main__":
#     import asyncio
#     # Note: Running top-level async requires asyncio.run()
#     # asyncio.run(example_main())
#     pass # Keep clean for import
