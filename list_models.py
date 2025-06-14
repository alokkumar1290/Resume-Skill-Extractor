import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

def list_available_models():
    # Load environment variables
    load_dotenv()
    
    # Get API key from environment
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("❌ Error: MISTRAL_API_KEY not found in .env file")
        return
    
    try:
        # Create API client
        api = HfApi(token=api_key)
        
        # List models from Mistral AI
        print("Fetching available models from Mistral AI...")
        models = api.list_models(author="mistralai")
        
        print("\nAvailable models:")
        for i, model in enumerate(models[:10], 1):  # Show first 10 models for brevity
            print(f"{i}. {model.modelId}")
            
        print("\nNote: If you don't see any models, your API key might not have access to private models.")
        
    except Exception as e:
        print(f"❌ Error fetching models: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Verify your MISTRAL_API_KEY in the .env file is correct")
        print("2. Check your internet connection")
        print("3. Make sure your Hugging Face account has access to Mistral models")
        print("4. Visit https://huggingface.co/mistralai to check available models")

if __name__ == "__main__":
    list_available_models()
