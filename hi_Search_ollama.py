import os
import logging
import numpy as np
import yaml
from hirag import HiRAG, QueryParam
import ollama  # Import the ollama library
from dataclasses import dataclass
from hirag.base import BaseKVStorage
from hirag._utils import compute_args_hash
import asyncio

# Load configuration from YAML file
# Ensure your config.yaml has an 'ollama' section with base_url, embedding_model, chat_model, embedding_dim
# and a 'model_params' section with max_token_size
try:
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    print("Error: config.yaml not found. Please create it with necessary ollama and model_params sections.")
    exit(1)
except KeyError as e:
    print(f"Error: Missing key in config.yaml: {e}. Ensure ollama and model_params sections are complete.")
    exit(1)


# Extract Ollama configurations
OLLAMA_EMBEDDING_MODEL = config['ollama']['embedding_model']
OLLAMA_CHAT_MODEL = config['ollama']['chat_model']
OLLAMA_URL = config['ollama'].get('base_url', 'http://localhost:11434') # Use default if not specified
OLLAMA_EMBEDDING_DIM = config['ollama']['embedding_dim']
MAX_TOKEN_SIZE = config['model_params']['max_token_size']

@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)

def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro

# Define the async Ollama client
# Note: Client instantiation might be better outside the functions if reused heavily,
# but keeping it simple here based on examples.
async def get_ollama_async_client():
    return ollama.AsyncClient(host=OLLAMA_URL)

@wrap_embedding_func_with_attrs(embedding_dim=OLLAMA_EMBEDDING_DIM, max_token_size=MAX_TOKEN_SIZE)
async def OLLAMA_embedding(texts: list[str]) -> np.ndarray:
    """Generates embeddings using the configured Ollama embedding model."""
    client = await get_ollama_async_client()
    embeddings = []
    for text in texts:
         # ollama.embed currently doesn't support batching in the library, process one by one
         # Keep an eye on library updates for potential batch support.
        response = await client.embed(model=OLLAMA_EMBEDDING_MODEL, input=text)
        embeddings.append(response['embedding'])
    return np.array(embeddings, dtype=np.float32)


async def OLLAMA_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """Sends a chat request to the configured Ollama chat model, using HiRAG cache."""
    client = await get_ollama_async_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if available-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages) # history_messages should already be in {"role": "...", "content": "..."} format
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None:
        # Use the specific Ollama chat model name for hashing
        args_hash = compute_args_hash(OLLAMA_CHAT_MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            logging.info(f"Cache hit for hash: {args_hash}")
            return if_cache_return["return"]
        logging.info(f"Cache miss for hash: {args_hash}")
    # -----------------------------------------------------

    # Ensure kwargs passed to ollama.chat are valid for its API
    # Filter out hashing_kv if it was passed initially
    valid_ollama_kwargs = {k: v for k, v in kwargs.items() if k != "hashing_kv"}

    response = await client.chat(
        model=OLLAMA_CHAT_MODEL, messages=messages, **valid_ollama_kwargs
    )

    response_content = response['message']['content']

    # Cache the response -----------------------------
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response_content, "model": OLLAMA_CHAT_MODEL}}
        )
        logging.info(f"Cached response for hash: {args_hash}")
    # -----------------------------------------------------
    return response_content


# Initialize HiRAG with Ollama functions
# Ensure hirag section in config.yaml exists and has necessary keys
try:
    graph_func = HiRAG(
        working_dir=config['hirag']['working_dir'],
        enable_llm_cache=config['hirag']['enable_llm_cache'],
        embedding_func=OLLAMA_embedding,
        best_model_func=OLLAMA_model_if_cache, # Use Ollama for both best and cheap
        cheap_model_func=OLLAMA_model_if_cache,
        enable_hierachical_mode=config['hirag']['enable_hierachical_mode'],
        embedding_batch_num=config['hirag']['embedding_batch_num'], # Consider Ollama's performance
        embedding_func_max_async=config['hirag']['embedding_func_max_async'], # Adjust based on Ollama setup
        enable_naive_rag=config['hirag']['enable_naive_rag']
    )
except KeyError as e:
    print(f"Error: Missing key in config.yaml under 'hirag': {e}")
    exit(1)

async def main():
    # --- Insertion Phase ---
    # Comment out this block if the working directory has already been indexed
    try:
        print("Attempting to insert data...")
        # Replace "your_data.txt" with the actual path to your text file
        file_path = "your_data.txt"
        with open(file_path, "r") as f:
            data = f.read()
        await graph_func.insert(data) # HiRAG insert is now async
        print(f"Data from {file_path} inserted successfully into {config['hirag']['working_dir']}")
    except FileNotFoundError:
        print(f"Error: Data file '{file_path}' not found. Please create it or comment out the insertion block.")
        # Decide if you want to exit or continue without insertion
        # exit(1)
        print("Continuing without data insertion.")
    except Exception as e:
        print(f"An error occurred during insertion: {e}")
        # exit(1) # Optional: exit if insertion fails


    # --- Query Phase ---
    print("\nPerforming hi search using Ollama:")
    query_text = "What are the key concepts discussed?" # Example query
    try:
        # HiRAG query is now async
        result = await graph_func.query(query_text, param=QueryParam(mode="hi"))
        print(f"\nQuery: {query_text}")
        print(f"Result:\n{result}")
    except Exception as e:
        print(f"\nAn error occurred during query: {e}")
        print("Ensure the Ollama server is running and models are available.")

if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    asyncio.run(main())
