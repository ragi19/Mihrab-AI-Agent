"""
Groq model implementations
"""

from typing import Any, AsyncIterator, Dict, List, Optional, Set

from groq import AsyncGroq

from ....core.message import Message, MessageRole
from ....core.types import ModelParameters, ModelResponse
from ....utils.logging import get_logger
from ...base import BaseModel, ModelCapability, ModelError, ModelInfo, TokenLimitError


class GroqModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        client: AsyncGroq,
        parameters: Optional[ModelParameters] = None,
    ):
        super().__init__(model_name, parameters)
        self.client = client
        self.provider_name = "groq"
        self.logger = get_logger("models.providers.groq")
        self.model_name = model_name
        self._capabilities = {ModelCapability.CHAT, ModelCapability.STREAMING}

        # Add function calling capability for supported models
        if "llama-3" in model_name.lower():
            self._capabilities.add(ModelCapability.FUNCTION_CALLING)

        # Set model context window and max tokens based on model name
        context_window = 8192  # Default
        max_tokens = 4096  # Default

        # Adjust context window based on model
        if "llama3-70b-8192" in model_name:
            context_window = 8192
        elif "llama2-70b-4096" in model_name:
            context_window = 4096
        elif "mixtral-8x7b-32768" in model_name:
            context_window = 32768
        elif "32k" in model_name:
            context_window = 32768
        elif "deepseek-r1-distill-llama-70b" in model_name:
            context_window = 32768
        elif "deepseek-coder-32b-instruct" in model_name:
            context_window = 32768
        elif "deepseek-coder-32b-instruct" in model_name:
            context_window = 32768

        # Create model info
        self.model_info = ModelInfo(
            id=model_name,
            name=model_name,
            provider=self.provider_name,
            capabilities=self._capabilities,
            context_window=context_window,
            max_tokens=max_tokens,
        )

    async def initialize(self) -> None:
        """Initialize the model

        This method is called after the model is created to perform any necessary setup.
        """
        # No initialization needed for Groq models
        pass

    @property
    def capabilities(self) -> Set[str]:
        """Get the capabilities of this model"""
        return self._capabilities

    async def generate(self, messages: List[Message], **kwargs: Any) -> Message:
        """Generate a response from the model"""
        try:
            return await self.generate_response(messages, **kwargs)
        except Exception as e:
            if "maximum token" in str(e).lower():
                raise TokenLimitError(str(e))
            raise ModelError(f"Groq API error: {str(e)}")

    async def generate_stream(
        self, messages: List[Message], **kwargs: Any
    ) -> AsyncIterator[Message]:
        """Stream a response from the model"""
        try:
            async for chunk in self.stream_response(messages):
                yield chunk
        except Exception as e:
            if "maximum token" in str(e).lower():
                raise TokenLimitError(str(e))
            raise ModelError(f"Groq API error: {str(e)}")

    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        """Update model parameters

        Args:
            parameters: Dictionary of parameters to update
        """
        if not parameters:
            return

        # Update the model's configuration with the provided parameters
        for key, value in parameters.items():
            self.set_config(key, value)

    async def generate_response(
        self, messages: List[Message], **kwargs: Any
    ) -> Message:
        """Generate a response from the model

        Args:
            messages: List of messages in the conversation
            **kwargs: Additional parameters to pass to the model

        Returns:
            Response message from the model
        """
        # Extract tools if provided
        tools = kwargs.pop("tools", None)

        try:
            # Convert messages to Groq format
            groq_messages = [msg.to_dict() for msg in messages]

            # Create completion request parameters
            params = {
                "model": self.model_name,
                "messages": groq_messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1024),
                "top_p": kwargs.get("top_p", 1.0),
                "stream": False,
            }

            # Add tools if provided and model supports function calling
            if tools and ModelCapability.FUNCTION_CALLING in self._capabilities:
                params["tools"] = tools

            # Create completion request
            completion = await self.client.chat.completions.create(**params)

            # Extract response content
            response_content = completion.choices[0].message.content

            # Create response message
            response = Message(role=MessageRole.ASSISTANT, content=response_content)

            # Check if there are tool calls in the response
            if (
                hasattr(completion.choices[0].message, "tool_calls")
                and completion.choices[0].message.tool_calls
            ):
                tool_calls = []
                for tool_call in completion.choices[0].message.tool_calls:
                    tool_calls.append(
                        {
                            "name": tool_call.function.name,
                            "parameters": tool_call.function.arguments,
                        }
                    )
                response.tool_calls = tool_calls

            return response

        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            raise

    async def stream_response(
        self, messages: List[Message], **kwargs: Any
    ) -> AsyncIterator[Message]:
        """Stream a response from the Groq API

        Args:
            messages: List of messages in the conversation
            **kwargs: Additional parameters to pass to the model

        Yields:
            Message chunks from the model
        """
        # Extract tools if provided
        tools = kwargs.pop("tools", None)

        # Create completion request parameters
        params = {
            "model": self.model_name,
            "messages": [msg.to_dict() for msg in messages],
            "stream": True,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1024),
            "top_p": kwargs.get("top_p", 1.0),
        }

        # Add tools if provided and model supports function calling
        if tools and ModelCapability.FUNCTION_CALLING in self._capabilities:
            params["tools"] = tools

        try:
            async for chunk in await self.client.chat.completions.create(**params):
                if chunk.choices[0].delta.content is not None:
                    yield Message(
                        role=MessageRole.ASSISTANT,
                        content=chunk.choices[0].delta.content,
                    )
        except Exception as e:
            self.logger.error(f"Error streaming response: {e}")
            raise

    async def count_tokens(self, text: str) -> int:
        """Estimate token count using tiktoken"""
        import tiktoken

        encoding = tiktoken.encoding_for_model(
            "gpt-3.5-turbo"
        )  # Use as fallback encoding
        return len(encoding.encode(text))
