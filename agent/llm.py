"""
Centralized LLM Configuration for Knowledge Graph Pipeline.

Simple, single-source configuration for Groq LLM client.
"""

import os
import logging
from typing import Dict, Any, List
from groq import Groq
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)





def get_model_name(config: Dict[str, Any] = None) -> str:
    """
    Get configured model name.
    
    Priority:
    1. Config dict 'llm_model' key
    2. GROQ_MODEL environment variable
    
    Args:
        config: Optional config dictionary
        
    Returns:
        Model name string
    """
    if config and 'llm_model' in config:
        return config['llm_model']
    
    return os.environ.get("GROQ_MODEL")


def get_temperature(config: Dict[str, Any] = None) -> float:
    """
    Get LLM temperature setting.
    
    Args:
        config: Optional config dictionary
        
    Returns:
        Temperature value (0.0 to 1.0)
    """
    if config and 'llm_temperature' in config:
        return float(config['llm_temperature'])
    
    return float(os.environ.get("LLM_TEMPERATURE", "0.0"))


def chat_completion(
    client: Groq,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 4000
) -> str:
    """
    Simple wrapper for Groq chat completion.
    
    Args:
        client: Groq client instance
        messages: List of message dicts with 'role' and 'content'
        model: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        
    Returns:
        Generated text content
    """
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        raise





def get_langchain_llm(config: Dict[str, Any] = None) -> ChatGroq:
    """
    Get LangChain-compatible Groq LLM for use with LangChain tools.
    
    Used by summarization and retrieval services.
    
    Args:
        config: Optional config dictionary
        
    Returns:
        ChatGroq instance compatible with LangChain tools
    """
    model = get_model_name(config)
    temperature = get_temperature(config)
    api_key = os.environ.get("GROQ_API_KEY")
    
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=api_key
    )


def get_llamaindex_llm(config: Dict[str, Any] = None):
    """
    Get LlamaIndex-compatible Groq LLM for use with LlamaIndex tools.
    
    This is useful for tools like PropertyGraphIndex that expect a LlamaIndex LLM.
    
    Args:
        config: Optional config dictionary
        
    Returns:
        Groq LLM instance compatible with LlamaIndex tools
    """
    from llama_index.llms.groq import Groq as LlamaGroq
    
    model = get_model_name(config)
    temperature = get_temperature(config)
    api_key = os.environ.get("GROQ_API_KEY")
    
    return LlamaGroq(
        model=model,
        temperature=temperature,
        api_key=api_key
    )
