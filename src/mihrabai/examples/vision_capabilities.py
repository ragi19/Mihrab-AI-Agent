"""
Example of using vision capabilities with supported models
"""

import asyncio
import base64
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from llm_agents import Message, MessageRole, create_agent
from llm_agents.models.base import ModelCapability

# Load environment variables from .env file
load_dotenv()


def encode_image_to_base64(image_path: str) -> Optional[str]:
    """
    Encode an image file to base64 string

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string or None if file not found
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None


async def run_vision_example(
    provider_name: str, model_name: str, image_path: str, prompt: str
):
    """
    Run a vision model example with the specified provider and model

    Args:
        provider_name: Name of the provider (e.g., "openai")
        model_name: Name of the vision-capable model
        image_path: Path to the image file
        prompt: Text prompt to accompany the image
    """
    print(f"\n=== Running Vision Model with {provider_name.upper()} ({model_name}) ===")

    # Encode the image to base64
    image_base64 = encode_image_to_base64(image_path)
    if not image_base64:
        print("Failed to encode image. Exiting example.")
        return

    # Create an agent with the specified provider and model
    try:
        agent = await create_agent(
            provider_name=provider_name,
            model_name=model_name,
            system_message="You are a helpful AI assistant with vision capabilities. Describe images accurately and answer questions about them.",
            required_capabilities={
                ModelCapability.VISION
            },  # Ensure model supports vision
        )
    except Exception as e:
        print(f"Error creating vision agent: {e}")
        print("This model may not support vision capabilities.")
        return

    # Create a message with image content
    message_content = {"type": "text", "text": prompt}

    # Add image data
    image_content = {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
    }

    # Create a message with both text and image
    message = Message(role=MessageRole.USER, content=[message_content, image_content])

    # Process the message and get a response
    try:
        print(f"User: {prompt} [IMAGE]")
        print("Processing image, please wait...")

        response = await agent.process_message(message)

        print(f"Assistant: {response.content}")
    except Exception as e:
        print(f"Error processing vision request: {e}")


async def main():
    """Run vision examples with supported providers"""
    # Example image path - adjust as needed
    # This should be a path to a local image file
    image_path = os.path.join(os.path.dirname(__file__), "sample_image.jpg")

    # Check if the image exists, if not create a placeholder message
    if not os.path.exists(image_path):
        print(f"Sample image not found at {image_path}")
        print(
            "Please place a sample image at this location or modify the path in the script."
        )

        # Create examples directory if it doesn't exist
        os.makedirs(os.path.dirname(image_path), exist_ok=True)

        # Provide instructions for adding an image
        print("\nTo use this example:")
        print(
            f"1. Add an image file named 'sample_image.jpg' to {os.path.dirname(image_path)}"
        )
        print("2. Run this script again")
        return

    # Example prompts for the image
    prompts = [
        "What's in this image? Please describe it in detail.",
        "What colors are prominent in this image?",
        "Is there any text visible in this image? If so, what does it say?",
    ]

    # Run with different providers that support vision
    if os.getenv("OPENAI_API_KEY"):
        # GPT-4 Vision models
        for prompt in prompts:
            await run_vision_example(
                "openai", "gpt-4-vision-preview", image_path, prompt
            )

    if os.getenv("ANTHROPIC_API_KEY"):
        # Claude 3 models with vision support
        for prompt in prompts:
            await run_vision_example(
                "anthropic", "claude-3-opus-20240229", image_path, prompt
            )

    if os.getenv("GROQ_API_KEY"):
        # Check if Groq has vision models available
        for prompt in prompts:
            await run_vision_example(
                "groq", "llama-3.2-11b-vision-preview", image_path, prompt
            )

    # If no API keys are available, use a fallback message
    if not any(
        [
            os.getenv("OPENAI_API_KEY"),
            os.getenv("ANTHROPIC_API_KEY"),
            os.getenv("GROQ_API_KEY"),
        ]
    ):
        print(
            "\nNo API keys found. Please set at least one of the following environment variables:"
        )
        print("- OPENAI_API_KEY")
        print("- ANTHROPIC_API_KEY")
        print("- GROQ_API_KEY")


if __name__ == "__main__":
    asyncio.run(main())
