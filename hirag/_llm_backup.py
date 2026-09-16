import numpy as np

from openai import AsyncOpenAI, AsyncAzureOpenAI, APIConnectionError, RateLimitError
import aiohttp
import json

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import os

from ._utils import compute_args_hash, wrap_embedding_func_with_attrs
from .base import BaseKVStorage

global_openai_async_client = None
global_azure_openai_async_client = None
global_deepseek_session = None
global_ollama_session = None


def get_openai_async_client_instance():
    global global_openai_async_client
    if global_openai_async_client is None:
        # Check for environment variables for custom OpenAI configuration
        base_url = os.environ.get("OPENAI_API_BASE", os.environ.get("OPENAI_BASE_URL", None))
        api_key = os.environ.get("OPENAI_API_KEY", 'ollama')  # Default to ollama if not set
        
        # Create the client with the environment variables if they exist
        if base_url:
            global_openai_async_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        else:
            global_openai_async_client = AsyncOpenAI()
    return global_openai_async_client


def get_azure_openai_async_client_instance():
    global global_azure_openai_async_client
    if global_azure_openai_async_client is None:
        # Check for environment variables for custom Azure configuration
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15")
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", None)
        azure_key = os.environ.get("AZURE_OPENAI_API_KEY", None)
        
        # If Azure configuration is available, use it
        if azure_endpoint and azure_key:
            global_azure_openai_async_client = AsyncAzureOpenAI(
                api_version=api_version,
                azure_endpoint=azure_endpoint,
                api_key=azure_key
            )
        else:
            # Fall back to the OpenAI client with Azure-compatible settings
            base_url = os.environ.get("OPENAI_API_BASE", os.environ.get("OPENAI_BASE_URL", None))
            api_key = os.environ.get("OPENAI_API_KEY", "ollama")
            
            if base_url:
                global_azure_openai_async_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
            else:
                global_azure_openai_async_client = AsyncOpenAI()
    return global_azure_openai_async_client


def get_deepseek_session():
    """Get or create a DeepSeek API session"""
    global global_deepseek_session
    if global_deepseek_session is None:
        global_deepseek_session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {os.environ.get('DEEPSEEK_API_KEY', '')}",
                "Content-Type": "application/json"
            }
        )
    return global_deepseek_session


def get_ollama_session():
    """Get or create an Ollama API session"""
    global global_ollama_session
    if global_ollama_session is None:
        ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Set up basic authentication if provided
        headers = {"Content-Type": "application/json"}
        auth = None
        if os.environ.get("OLLAMA_API_KEY"):
            headers["Authorization"] = f"Bearer {os.environ.get('OLLAMA_API_KEY')}"
            
        global_ollama_session = aiohttp.ClientSession(
            base_url=ollama_base_url,
            headers=headers,
            auth=auth
        )
    return global_ollama_session


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = get_openai_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content


async def gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    model = os.environ.get("OPENAI_MODEL", "gpt-4o")
    return await openai_complete_if_cache(
        model,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

async def gpt_35_turbo_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # Use a model from env if available, otherwise fallback to GPT-3.5
    model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
    return await openai_complete_if_cache(
        model,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # Use a model from env if available, otherwise fallback to GPT-4o-mini
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    return await openai_complete_if_cache(
        model,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

async def gpt_custom_model_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # Get model name from environment variables or fallback to a default
    model_name = os.environ.get("OPENAI_MODEL_NAME", os.environ.get("OPENAI_MODEL", "llama3"))
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
async def deepseek_complete_if_cache(
    model=None, prompt=None, system_prompt=None, history_messages=[], **kwargs
) -> str:
    session = get_deepseek_session()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    
    # Get model name from environment variables or use default
    if model is None:
        model = os.environ.get("DEEPSEEK_MODEL", os.environ.get("OPENAI_MODEL", "deepseek-chat"))
    
    # Prepare messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    # Check cache if available
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    
    # Prepare request
    payload = {
        "model": model,
        "messages": messages,
    }
    
    # Add additional parameters
    for key, value in kwargs.items():
        if key in ["temperature", "top_p", "max_tokens", "stream"]:
            payload[key] = value
    
    # Make API request
    deepseek_base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    async with session.post(f"{deepseek_base_url}/v1/chat/completions", json=payload) as response:
        if response.status != 200:
            error_text = await response.text()
            raise RuntimeError(f"DeepSeek API error: {response.status} - {error_text}")
        
        response_json = await response.json()
        completion = response_json["choices"][0]["message"]["content"]
        
        # Cache the response if enabled
        if hashing_kv is not None:
            await hashing_kv.upsert(
                {args_hash: {"return": completion, "model": model}}
            )
            await hashing_kv.index_done_callback()
        
        return completion


async def deepseek_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # Model can be overridden with OPENAI_MODEL_NAME for consistency with other API calls
    model = os.environ.get("OPENAI_MODEL_NAME", os.environ.get("DEEPSEEK_MODEL", os.environ.get("OPENAI_MODEL", "deepseek-chat")))
    return await deepseek_complete_if_cache(
        model,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
async def ollama_complete_if_cache(
    model=None, prompt=None, system_prompt=None, history_messages=[], **kwargs
) -> str:
    session = get_ollama_session()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    
    # Get model name from environment variables or use default
    if model is None:
        model = os.environ.get("OPENAI_MODEL_NAME", os.environ.get("OPENAI_MODEL", os.environ.get("GLM_MODEL", "llama3")))
    
    # Prepare messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    # Check cache if available
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    
    # Prepare request
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    
    # Add additional parameters
    for key, value in kwargs.items():
        if key in ["temperature", "top_p", "num_predict"]:
            payload[key] = value
    
    # Make API request
    async with session.post("/api/chat", json=payload) as response:
        if response.status != 200:
            error_text = await response.text()
            raise RuntimeError(f"Ollama API error: {response.status} - {error_text}")
        
        response_json = await response.json()
        completion = response_json["message"]["content"]
        
        # Cache the response if enabled
        if hashing_kv is not None:
            await hashing_kv.upsert(
                {args_hash: {"return": completion, "model": model}}
            )
            await hashing_kv.index_done_callback()
        
        return completion


async def ollama_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    model = os.environ.get("OPENAI_MODEL_NAME", os.environ.get("OPENAI_MODEL", os.environ.get("GLM_MODEL", "llama3")))
    return await ollama_complete_if_cache(
        model,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=3584, max_token_size=8192)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_embedding(texts: list[str]) -> np.ndarray:
    openai_async_client = get_openai_async_client_instance()
    # Use model from env if available
    model = os.environ.get("OPENAI_EMBEDDING_MODEL", os.environ.get("OPENAI_MODEL", "text-embedding-3-small"))
    response = await openai_async_client.embeddings.create(
        model=model, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_complete_if_cache(
    deployment_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    azure_openai_client = get_azure_openai_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(deployment_name, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await azure_openai_client.chat.completions.create(
        model=deployment_name, messages=messages, **kwargs
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {
                args_hash: {
                    "return": response.choices[0].message.content,
                    "model": deployment_name,
                }
            }
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content


async def azure_gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # Use model from env if available
    model = os.environ.get("OPENAI_MODEL", "gpt-4o")
    return await azure_openai_complete_if_cache(
        model,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def azure_gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # Use model from env if available
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    return await azure_openai_complete_if_cache(
        model,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

async def azure_openai_custom_model_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # Get model name from environment variables or fallback to a default
    model_name = os.environ.get("OPENAI_MODEL_NAME", os.environ.get("OPENAI_MODEL", "llama3"))
    return await azure_openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=3584, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_embedding(texts: list[str]) -> np.ndarray:
    azure_openai_client = get_azure_openai_async_client_instance()
    # Use model from env if available
    model = os.environ.get("OPENAI_EMBEDDING_MODEL", os.environ.get("OPENAI_MODEL", "text-embedding-3-small"))
    response = await azure_openai_client.embeddings.create(
        model=model, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


@wrap_embedding_func_with_attrs(embedding_dim=3584, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
async def deepseek_embedding(texts: list[str]) -> np.ndarray:
    session = get_deepseek_session()
    
    # Get embedding model from environment variables
    model = os.environ.get("DEEPSEEK_EMBEDDING_MODEL", os.environ.get("OPENAI_MODEL", "deepseek-embedding"))
    
    # Prepare request payload
    payload = {
        "model": model,
        "input": texts,
        "encoding_format": "float"
    }
    
    # Make API request
    deepseek_base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    async with session.post(f"{deepseek_base_url}/v1/embeddings", json=payload) as response:
        if response.status != 200:
            error_text = await response.text()
            raise RuntimeError(f"DeepSeek Embedding API error: {response.status} - {error_text}")
        
        response_json = await response.json()
        embeddings = [data["embedding"] for data in response_json["data"]]
        
        return np.array(embeddings)


@wrap_embedding_func_with_attrs(embedding_dim=3584, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
async def ollama_embedding(texts: list[str]) -> np.ndarray:
    session = get_ollama_session()
    
    # Get embedding model from environment variables
    # Ollama might use the same model for completion and embedding
    model = os.environ.get("OLLAMA_EMBEDDING_MODEL", os.environ.get("OPENAI_MODEL", os.environ.get("GLM_MODEL", "llama3")))
    
    # Ollama can only process one text at a time for embeddings
    all_embeddings = []
    for text in texts:
        # Prepare request payload
        payload = {
            "model": model,
            "prompt": text,
        }
        
        # Make API request
        async with session.post("/api/embeddings", json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Ollama Embedding API error: {response.status} - {error_text}")
            
            response_json = await response.json()
            embeddings = response_json["embedding"]
            all_embeddings.append(embeddings)
    
    return np.array(all_embeddings)
