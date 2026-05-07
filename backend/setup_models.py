#!/usr/bin/env python3
"""
Setup script to download all pretrained models
Run this once before starting the server
"""
import os
import subprocess
import sys

def download_models():
    """Download only essential models for free tier"""
    
    print("🚀 Setting up pretrained models for AI SkillFit...")
    print("=" * 60)
    
    # 1. Whisper (essential)
    print("\n📥 Downloading Whisper model...")
    import whisper
    whisper.load_model("base")
    print("✅ Whisper model ready")
    
    # 2. Sentence Transformers (essential)
    print("\n📥 Downloading Sentence Transformer...")
    from sentence_transformers import SentenceTransformer
    SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Sentence Transformer ready")
    
    # Skip heavy models to save memory
    print("\n⚠️  Skipping heavy models to save memory on free tier")
    print("✅ Essential models ready")
    
    print("\n" + "=" * 60)
    print("✨ Essential models downloaded successfully!")
    print("🎯 Running in lightweight mode for free tier")
    print("\nYou can now start the server with:")
    print("  python main.py")
    print("  or")
    print("  uvicorn main:app --reload")

if __name__ == "__main__":
    try:
        download_models()
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        print("\nPlease ensure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
