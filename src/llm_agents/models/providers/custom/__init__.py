"""
Template for implementing custom model providers
"""

from typing import Any, Dict, Set

from ...types import ModelCapability, ModelInfo
from ..base import BaseProvider, ProviderError
from .models import CustomModel


class CustomProvider(BaseProvider):
    """Template for implementing custom model providers

    Example usage:
    ```python
    class MyCustomProvider(CustomProvider):
        SUPPORTED_MODELS = {
            "my-model": ModelInfo(
                name="my-model",
                capabilities={
                    ModelCapability.CHAT,
                    ModelCapability.COMPLETION
                },
                max_tokens=4096,
                context_window=4096
            )
        }

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.client = MyCustomClient(kwargs["api_key"])

            for model_name, model_info in self.SUPPORTED_MODELS.items():
                self.register_model(model_name, MyCustomModel, model_info)
    ```
    """

    # Define your supported models and their capabilities
    SUPPORTED_MODELS: Dict[str, ModelInfo] = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initialize your client/API here
        if "api_key" not in kwargs:
            raise ProviderError("API key required")

        # Register supported models
        for model_name, model_info in self.SUPPORTED_MODELS.items():
            self.register_model(model_name, CustomModel, model_info)

    async def create_model(self, model_name: str) -> CustomModel:
        """Create a model instance

        Args:
            model_name: Name of the model to create

        Returns:
            Initialized model instance

        Raises:
            ProviderError: If model creation fails
        """
        if model_name not in self._models:
            raise ProviderError(f"Model {model_name} not supported")

        model = CustomModel(
            model_name=model_name,
            client=None,  # Add your client here
            parameters=self.get_default_parameters(model_name),
        )

        return model

    def register_model(
        self, model_name: str, model_class: type, model_info: ModelInfo
    ) -> None:
        """Register a model with this provider"""
        self._models[model_name] = model_class
        self._model_info[model_name] = model_info
        self.logger.info(f"Registered custom model: {model_name}")

    def _get_model_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for model initialization"""
        return {
            "client": None,  # Add your client here
            "parameters": self._config.get("default_parameters", {}),
        }

    @classmethod
    def get_required_config_keys(cls) -> Set[str]:
        """Get required configuration keys"""
        # Add any additional required config keys
        return {"api_key"}

    async def validate_api_key(self) -> bool:
        """Validate the API key

        Returns:
            True if API key is valid

        Raises:
            ProviderError: If validation fails
        """
        raise NotImplementedError("API key validation must be implemented")
