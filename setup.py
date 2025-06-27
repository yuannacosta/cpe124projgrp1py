import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("Installing required packages")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def download_models():
    """Download required models"""
    print("Downloading models")
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sentence_transformers import SentenceTransformer
    
    print("Downloading DialoGPT model")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    
    print("Downloading embedding model")
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    print("Setup complete!")

def create_directories():
    """Create necessary directories"""
    os.makedirs("chroma_db", exist_ok=True)
    os.makedirs("models", exist_ok=True)

if __name__ == "__main__":
    print("Setting up Binondo Heritage RAG Chatbot")
    
    create_directories()
    install_requirements()
    download_models()
    
    print("\n Setup complete!")
    print("Run the app with: streamlit run app.py")
