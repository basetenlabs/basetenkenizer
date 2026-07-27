from ._native import Tokenizer as Tokenizer
from .tiktoken import (
    tiktoken_model_to_tokenizer_json as tiktoken_model_to_tokenizer_json,
    tiktoken_to_tokenizer_json as tiktoken_to_tokenizer_json,
)

__all__: list[str]

def patch_transformers(apply_chat_template: bool = False) -> None:
    """Patch Transformers to use basetenkenizer for encoding."""
    ...

def unpatch_transformers() -> None:
    """Restore the Transformers objects replaced by patch_transformers."""
    ...
