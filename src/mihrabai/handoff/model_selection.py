"""
Model selection and fallback mechanisms for handoff agents

This module provides functionality for dynamically selecting models
and implementing fallback strategies when a model is unavailable.
"""

import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging
import hashlib
import json

from ..models.base import BaseModel, ModelCapability, ModelError
from ..models.factory import create_model, ModelCreationError, ModelFactory
from ..utils.logging import get_logger

logger = get_logger("handoff.model_selection")


class ModelSelectionStrategy:
    """Base class for model selection strategies"""

    async def select_model(
        self,
        model_candidates: List[Dict[str, Any]],
        required_capabilities: Optional[Set[str]] = None,
    ) -> Tuple[Optional[BaseModel], Dict[str, Any]]:
        """
        Select a model from the list of candidates

        Args:
            model_candidates: List of model configurations to try
            required_capabilities: Optional set of required capabilities

        Returns:
            Tuple of (selected model, model config) or (None, {}) if no model could be selected
        """
        raise NotImplementedError("Subclasses must implement select_model")


class PrioritizedModelStrategy(ModelSelectionStrategy):
    """
    Strategy that tries models in priority order

    This strategy attempts to create models in the order they are provided,
    falling back to the next model if creation fails.
    """

    async def select_model(
        self,
        model_candidates: List[Dict[str, Any]],
        required_capabilities: Optional[Set[str]] = None,
    ) -> Tuple[Optional[BaseModel], Dict[str, Any]]:
        """
        Select a model from the list of candidates in priority order

        Args:
            model_candidates: List of model configurations to try
            required_capabilities: Optional set of required capabilities

        Returns:
            Tuple of (selected model, model config) or (None, {}) if no model could be selected
        """
        if not model_candidates:
            logger.warning("No model candidates provided")
            return None, {}

        # Convert string capabilities to ModelCapability enum if needed
        capabilities_set = None
        if required_capabilities:
            capabilities_set = set()
            for cap in required_capabilities:
                if isinstance(cap, str):
                    try:
                        capabilities_set.add(getattr(ModelCapability, cap.upper()))
                    except AttributeError:
                        logger.warning(f"Unknown capability: {cap}")
                else:
                    capabilities_set.add(cap)

        # Try each model in order
        for model_config in model_candidates:
            model_name = model_config.get("model_name")
            provider_name = model_config.get("provider_name")
            provider_kwargs = model_config.get("provider_kwargs", {})
            model_parameters = model_config.get("model_parameters", {})

            if not model_name:
                logger.warning("Model configuration missing model_name")
                continue

            try:
                logger.info(f"Attempting to create model {model_name} with provider {provider_name}")
                model = await create_model(
                    model_name=model_name,
                    provider_name=provider_name,
                    provider_kwargs=provider_kwargs,
                    model_parameters=model_parameters,
                    required_capabilities=capabilities_set,
                )
                logger.info(f"Successfully created model {model_name}")
                return model, model_config
            except (ModelCreationError, ModelError) as e:
                logger.warning(f"Failed to create model {model_name}: {e}")
                continue

        logger.error("All model candidates failed")
        return None, {}


class CapabilityBasedStrategy(ModelSelectionStrategy):
    """
    Strategy that selects models based on capabilities

    This strategy filters models by required capabilities and then
    selects the best match based on additional criteria.
    """

    async def select_model(
        self,
        model_candidates: List[Dict[str, Any]],
        required_capabilities: Optional[Set[str]] = None,
    ) -> Tuple[Optional[BaseModel], Dict[str, Any]]:
        """
        Select a model based on capabilities and other criteria

        Args:
            model_candidates: List of model configurations to try
            required_capabilities: Optional set of required capabilities

        Returns:
            Tuple of (selected model, model config) or (None, {}) if no model could be selected
        """
        if not model_candidates:
            logger.warning("No model candidates provided")
            return None, {}

        # Convert string capabilities to ModelCapability enum if needed
        capabilities_set = None
        if required_capabilities:
            capabilities_set = set()
            for cap in required_capabilities:
                if isinstance(cap, str):
                    try:
                        capabilities_set.add(getattr(ModelCapability, cap.upper()))
                    except AttributeError:
                        logger.warning(f"Unknown capability: {cap}")
                else:
                    capabilities_set.add(cap)

        # Filter models by required capabilities
        filtered_candidates = []
        for model_config in model_candidates:
            model_capabilities = model_config.get("capabilities", set())
            if not capabilities_set or all(cap in model_capabilities for cap in capabilities_set):
                filtered_candidates.append(model_config)

        if not filtered_candidates:
            logger.warning("No models match the required capabilities")
            return None, {}

        # Sort by priority if available, otherwise use the first match
        sorted_candidates = sorted(
            filtered_candidates,
            key=lambda x: x.get("priority", 0),
            reverse=True,
        )

        # Try each model in priority order
        for model_config in sorted_candidates:
            model_name = model_config.get("model_name")
            provider_name = model_config.get("provider_name")
            provider_kwargs = model_config.get("provider_kwargs", {})
            model_parameters = model_config.get("model_parameters", {})

            try:
                logger.info(f"Attempting to create model {model_name} with provider {provider_name}")
                model = await create_model(
                    model_name=model_name,
                    provider_name=provider_name,
                    provider_kwargs=provider_kwargs,
                    model_parameters=model_parameters,
                    required_capabilities=capabilities_set,
                )
                logger.info(f"Successfully created model {model_name}")
                return model, model_config
            except (ModelCreationError, ModelError) as e:
                logger.warning(f"Failed to create model {model_name}: {e}")
                continue

        logger.error("All model candidates failed")
        return None, {}


class RoundRobinStrategy(ModelSelectionStrategy):
    """Strategy that cycles through available models in a round-robin fashion."""
    
    def __init__(self):
        self._last_index = -1
        self._failed_attempts = {}
    
    async def select_model(
        self, 
        model_candidates: List[Dict[str, Any]], 
        required_capabilities: Optional[Set[str]] = None
    ) -> Tuple[Optional[BaseModel], Optional[Dict[str, Any]]]:
        """
        Select a model from the candidates in a round-robin fashion.
        
        Args:
            model_candidates: List of model configuration dictionaries
            required_capabilities: Set of required model capabilities
            
        Returns:
            Tuple of (selected model, model configuration) or (None, None) if no model could be selected
        """
        if not model_candidates:
            return None, None
        
        # Filter by capabilities if required
        if required_capabilities:
            filtered_candidates = []
            for candidate in model_candidates:
                candidate_capabilities = set(candidate.get("capabilities", []))
                if required_capabilities.issubset(candidate_capabilities):
                    filtered_candidates.append(candidate)
            
            if not filtered_candidates:
                logger.warning(f"No models found with required capabilities: {required_capabilities}")
                return None, None
            
            candidates = filtered_candidates
        else:
            candidates = model_candidates
        
        # Reset failed attempts if we've tried all models
        if len(self._failed_attempts) >= len(candidates):
            self._failed_attempts = {}
        
        # Try each model in sequence, starting from the next one after the last used
        start_index = (self._last_index + 1) % len(candidates)
        current_index = start_index
        
        while True:
            candidate = candidates[current_index]
            model_key = f"{candidate.get('provider_name')}:{candidate.get('model_name')}"
            
            # Skip if we've already tried this model and it failed
            if model_key in self._failed_attempts:
                current_index = (current_index + 1) % len(candidates)
                if current_index == start_index:
                    # We've tried all models and none worked
                    return None, None
                continue
            
            try:
                model_name = candidate.get("model_name")
                provider_name = candidate.get("provider_name")
                provider_kwargs = candidate.get("provider_kwargs", {})
                model_params = candidate.get("model_params", {})
                
                model = await ModelFactory.create_model(
                    model_name=model_name,
                    provider_name=provider_name,
                    provider_kwargs=provider_kwargs,
                    **model_params
                )
                
                # Update the last index
                self._last_index = current_index
                
                return model, candidate
            except ModelCreationError as e:
                logger.warning(f"Failed to create model {candidate.get('model_name')}: {e}")
                # Mark this model as failed
                self._failed_attempts[model_key] = True
                
                # Try the next model
                current_index = (current_index + 1) % len(candidates)
                if current_index == start_index:
                    # We've tried all models and none worked
                    return None, None
        
        return None, None


class ModelSelectionManager:
    """
    Manager for model selection and fallback

    This class provides methods for selecting models based on different strategies
    and implementing fallback mechanisms when a model is unavailable.
    """

    def __init__(self, default_strategy: Optional[ModelSelectionStrategy] = None):
        """
        Initialize the model selection manager

        Args:
            default_strategy: Default strategy to use for model selection
        """
        self.default_strategy = default_strategy or PrioritizedModelStrategy()
        self.strategies = {
            "prioritized": PrioritizedModelStrategy(),
            "capability_based": CapabilityBasedStrategy(),
            "round_robin": RoundRobinStrategy(),
        }
        self.model_cache = {}  # Cache for created models

    async def select_model(
        self,
        model_candidates: List[Dict[str, Any]],
        strategy_name: Optional[str] = None,
        required_capabilities: Optional[Set[str]] = None,
        use_cache: bool = True,
    ) -> Tuple[Optional[BaseModel], Dict[str, Any]]:
        """
        Select a model using the specified strategy

        Args:
            model_candidates: List of model configurations to try
            strategy_name: Name of the strategy to use (default: use default_strategy)
            required_capabilities: Optional set of required capabilities
            use_cache: Whether to use cached models

        Returns:
            Tuple of (selected model, model config) or (None, {}) if no model could be selected
        """
        # Check cache first if enabled
        if use_cache:
            cache_key = self._get_cache_key(model_candidates, required_capabilities)
            if cache_key in self.model_cache:
                logger.info(f"Using cached model for {cache_key}")
                return self.model_cache[cache_key]

        # Select strategy
        strategy = self.default_strategy
        if strategy_name and strategy_name in self.strategies:
            strategy = self.strategies[strategy_name]

        # Select model
        model, config = await strategy.select_model(model_candidates, required_capabilities)

        # Cache result if successful and caching is enabled
        if model is not None and use_cache:
            cache_key = self._get_cache_key(model_candidates, required_capabilities)
            self.model_cache[cache_key] = (model, config)

        return model, config

    def _get_cache_key(
        self, model_candidates: List[Dict[str, Any]], capabilities: Optional[Set[str]] = None
    ) -> str:
        """
        Generate a cache key for the given model candidates and capabilities

        Args:
            model_candidates: List of model configurations
            capabilities: Optional set of required capabilities

        Returns:
            Cache key string
        """
        # Create a simple hash of the model candidates and capabilities
        models_str = ",".join(
            [f"{m.get('model_name')}:{m.get('provider_name', '')}" for m in model_candidates]
        )
        caps_str = ",".join(sorted([str(c) for c in (capabilities or [])]))
        return f"{models_str}|{caps_str}"

    def register_strategy(self, name: str, strategy: ModelSelectionStrategy) -> None:
        """
        Register a new model selection strategy

        Args:
            name: Name of the strategy
            strategy: Strategy instance
        """
        self.strategies[name] = strategy

    def set_default_strategy(self, strategy_name: str) -> None:
        """
        Set the default strategy

        Args:
            strategy_name: Name of the strategy to use as default
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        self.default_strategy = self.strategies[strategy_name]

    def clear_cache(self) -> None:
        """Clear the model cache"""
        self.model_cache = {}


# Default model selection manager instance
default_manager = ModelSelectionManager()


async def select_model(
    model_candidates: List[Dict[str, Any]],
    strategy_name: Optional[str] = None,
    required_capabilities: Optional[Set[str]] = None,
    use_cache: bool = True,
) -> Tuple[Optional[BaseModel], Dict[str, Any]]:
    """
    Select a model using the default manager

    Args:
        model_candidates: List of model configurations to try
        strategy_name: Name of the strategy to use
        required_capabilities: Optional set of required capabilities
        use_cache: Whether to use cached models

    Returns:
        Tuple of (selected model, model config) or (None, {}) if no model could be selected
    """
    return await default_manager.select_model(
        model_candidates, strategy_name, required_capabilities, use_cache
    )


def create_model_config(
    model_name: str,
    provider_name: Optional[str] = None,
    provider_kwargs: Optional[Dict[str, Any]] = None,
    model_parameters: Optional[Dict[str, Any]] = None,
    priority: int = 0,
    capabilities: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Create a model configuration dictionary

    Args:
        model_name: Name of the model
        provider_name: Name of the provider
        provider_kwargs: Additional arguments for provider initialization
        model_parameters: Model-specific parameters
        priority: Priority of this model (higher values = higher priority)
        capabilities: Set of capabilities this model supports

    Returns:
        Model configuration dictionary
    """
    return {
        "model_name": model_name,
        "provider_name": provider_name,
        "provider_kwargs": provider_kwargs or {},
        "model_parameters": model_parameters or {},
        "priority": priority,
        "capabilities": capabilities or set(),
    } 