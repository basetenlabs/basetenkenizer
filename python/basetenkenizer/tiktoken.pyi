from pathlib import Path
from typing import Any, Mapping

def tiktoken_model_to_tokenizer_json(
    model: str | Path,
    *,
    pattern: str | None = None,
    special_tokens: Mapping[str, int] | None = None,
    pretty: bool = False,
) -> str | None:
    """Convert a tiktoken BPE model into Hugging Face tokenizer JSON."""
    ...

def tiktoken_to_tokenizer_json(
    encoding: Any,
    *,
    pretty: bool = False,
) -> str | None:
    """Convert a tiktoken encoding or encoding name into tokenizer JSON."""
    ...
