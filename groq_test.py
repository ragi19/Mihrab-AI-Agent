"""
Simple Groq API Test

This script tests the connection to the Groq API with a valid API key.
"""

import os
import sys
from groq import Groq

def main():
    # Get API key from command line argument or environment variable
    api_key = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        print("Error: Please provide a Groq API key as a command line argument or set the GROQ_API_KEY environment variable.")
        sys.exit(1)
    
    # Initialize Groq client
    client = Groq(api_key=api_key)
    
    try:
        # Test the API key with a simple request
        response = client.chat.completions.create(
            model="llama3-8b-8192",  # Using a smaller model for a quick test
            messages=[{"role": "user", "content": "Hello, world!"}],
            max_tokens=10
        )
        
        # Print the response
        print("API Key is valid!")
        print(f"Response: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 