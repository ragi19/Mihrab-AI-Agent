"""
Model provider implementations package
"""

from .anthropic.models import ClaudeModel
from .anthropic.provider import AnthropicProvider
from .base import BaseProvider, ProviderError
from .custom.models import CustomModel

# Make sure to import relevant providers
try:
    from .openai.models import GPTModel
    from .openai.provider import OpenAIProvider
except ImportError:
    pass

try:
    from .groq.models import GroqModel
    from .groq.provider import GroqProvider
except ImportError:
    pass

try:
    from .grok.models import GrokModel
except ImportError:
    pass

__all__ = [
    "BaseProvider",
    "ProviderError",
    "AnthropicProvider",
    "ClaudeModel",
    "CustomModel",
]

# Add optional providers to __all__ if available
try:
    __all__.extend(["OpenAIProvider", "GPTModel"])
except NameError:
    pass

try:
    __all__.extend(["GroqProvider", "GroqModel"])
except NameError:
    pass

try:
    __all__.append("GrokModel")
except NameError:
    pass
