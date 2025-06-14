import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

def test_hf_api():
    # Load environment variables
    load_dotenv()
    
    # Get API key and model from environment
    api_key = os.getenv("HF_API_KEY")  # Changed from MISTRAL_API_KEY to standard HF_API_KEY
    model = os.getenv("HF_MODEL") or "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Added default model
    
    # Remove any quotes from the model name
    if model:
        model = model.strip('"\'')
    
    print(f"Using model: {model}")
    
    if not api_key:
        print("❌ Error: HF_API_KEY not found in .env file")
        return False
    
    # Updated list of available models that work with Inference API
    available_models = [
        model,  # Try the configured model first
        "mistralai/Mixtral-8x7B-Instruct-v0.1",  # Currently popular Mistral model
        "mistralai/Mistral-7B-Instruct-v0.1",
        "google/gemma-7b-it",  # Good alternative
        "meta-llama/Llama-2-7b-chat-hf",  # Another good alternative
        "HuggingFaceH4/zephyr-7b-beta"  # Lightweight option
    ]
    
    for current_model in available_models:
        try:
            print(f"\nTrying model: {current_model}")
            print("Creating InferenceClient...")
            # Initialize client with model directly
            client = InferenceClient(model=current_model, token=api_key)
            
            print("Sending test request...")
            # Simplified prompt with shorter expected response
            response = client.text_generation(
                prompt="Respond with 'OK' if you're working.",
                max_new_tokens=10,
                temperature=0.7  # Added temperature for more varied responses
            )
            
            print(f"✅ API Response: {response}")
            
            # If we get here, the request was successful
            print(f"\n✅ Successfully connected using model: {current_model}")
            print(f"✅ Please update your .env file with: HF_MODEL=\"{current_model}\"")
            return True
            
        except Exception as e:
            print(f"❌ Error with model {current_model}: {str(e)}")
            continue
    
    # If we get here, all models failed
    print("\n❌ Failed to connect with any model")
    print("\nTroubleshooting steps:")
    print("1. Verify your HF_API_KEY in the .env file is correct")
    print("2. Check your Hugging Face account has access to these models")
    print("3. Some models require Pro account: https://huggingface.co/pricing")
    print("4. Visit https://huggingface.co/models to check available models")
    print("5. Try reducing the max_new_tokens parameter")
    return False

if __name__ == "__main__":
    print("Testing Hugging Face API connection...")
    success = test_hf_api()
    if success:
        print("✅ API connection successful!")
    else:
        print("❌ Failed to connect to the API. Please check the troubleshooting steps above.")