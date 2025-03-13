"""
Model provider implementations package
"""
from .base import BaseProvider, ProviderError
from .anthropic.provider import AnthropicProvider
from .anthropic.models import ClaudeModel
from .custom.models import CustomModel

# Make sure to import relevant providers
try:
    from .openai.provider import OpenAIProvider
    from .openai.models import GPTModel
except ImportError:
    pass

try:
    from .groq.provider import GroqProvider
    from .groq.models import GroqModel
except ImportError:
    pass

try:
    from .grok.models import GrokModel
except ImportError:
    pass

__all__ = [
    'BaseProvider',
    'ProviderError',
    'AnthropicProvider',
    'ClaudeModel',
    'CustomModel'
]

# Add optional providers to __all__ if available
try:
    __all__.extend(['OpenAIProvider', 'GPTModel'])
except NameError:
    pass

try:
    __all__.extend(['GroqProvider', 'GroqModel'])
except NameError:
    pass

try:
    __all__.append('GrokModel')
except NameError:
    pass