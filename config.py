import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Configuration for the LLM model"""
    model_name: str = "microsoft/DialoGPT-medium"
    max_length: int = 512
    temperature: float = 0.7
    do_sample: bool = True
    device: str = "auto"

@dataclass
class EmbeddingConfig:
    """Configuration for embeddings"""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200

@dataclass
class VectorStoreConfig:
    """Configuration for vector store"""
    persist_directory: str = "./chroma_db"
    collection_name: str = "binondo_heritage"
    search_k: int = 3

@dataclass
class AppConfig:
    """Main application configuration"""
    page_title: str = "🏮 Binondo Heritage Guide"
    page_icon: str = "🏮"
    layout: str = "wide"

    model: ModelConfig = ModelConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    vectorstore: VectorStoreConfig = VectorStoreConfig()

ALTERNATIVE_MODELS = {
    "small": "microsoft/DialoGPT-small",  
    "large": "microsoft/DialoGPT-large",  
    "llama": "huggingface/CodeLlama-7b-Python-hf", 
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",  
    "phi": "microsoft/phi-2", 

def get_config() -> AppConfig:
    """Get application configuration"""
    return AppConfig()
