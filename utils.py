import streamlit as st
import torch
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_gpu_availability() -> Dict[str, Any]:
    """Check GPU availability and return system info"""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
    }
    return info

def format_chat_message(message: str, is_user: bool = False) -> str:
    """Format chat message for display"""
    if is_user:
        return f"**You:** {message}"
    else:
        return f"**Heritage Guide:** {message}"

def get_suggested_questions() -> List[str]:
    """Get list of suggested questions"""
    return [
        "Tell me about the history of Binondo Church",
        "What are the must-visit heritage sites in Binondo?",
        "What traditional foods should I try in Binondo?",
        "How did Binondo become the world's oldest Chinatown?",
        "What cultural festivals happen in Binondo?",
        "Where can I find traditional Chinese medicine shops?",
        "What's the significance of Escolta Street?",
        "Tell me about Saint Lorenzo Ruiz",
        "What are some traditional Chinese-Filipino dishes?",
        "How has Binondo preserved its heritage over 400+ years?"
    ]

def display_system_info():
    """Display system information in sidebar"""
    gpu_info = check_gpu_availability()
    
    st.sidebar.markdown("### 🖥️ System Information")
    
    if gpu_info["cuda_available"]:
        st.sidebar.success(f"🚀 GPU: {gpu_info['device_name']}")
        st.sidebar.info(f"📊 GPU Count: {gpu_info['device_count']}")
    else:
        st.sidebar.warning("💻 Running on CPU")
    
    # Memory usage (if available)
    if gpu_info["cuda_available"]:
        try:
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            st.sidebar.info(f"🧠 GPU Memory: {memory_allocated:.1f}GB / {memory_reserved:.1f}GB")
        except:
            pass

def create_download_link(text: str, filename: str) -> str:
    """Create a download link for text content"""
    import base64
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:text/plain;base64,{b64}" download="{filename}">Download {filename}</a>'

def log_user_interaction(user_input: str, bot_response: str):
    """Log user interactions for analytics (optional)"""
    logger.info(f"User Query: {user_input[:100]}...")
    logger.info(f"Bot Response Length: {len(bot_response)} characters")

class StreamlitLogger:
    """Custom logger for Streamlit apps"""
    
    @staticmethod
    def info(message: str):
        st.info(f"ℹ️ {message}")
    
    @staticmethod
    def success(message: str):
        st.success(f"✅ {message}")
    
    @staticmethod
    def warning(message: str):
        st.warning(f"⚠️ {message}")
    
    @staticmethod
    def error(message: str):
        st.error(f"❌ {message}")
