# Baseten Tokenizer

High-performance, Rust-backed BPE tokenization for inference. The Python
package is named `basetenkenizer`.

## Performance

![Kimi K3 tokenization latency comparison](assets/basetenkenizer_vs_gigatokenfastokenstiktoken.png)

## Install

```bash
pip install basetenkenizer
```

Loading by Hugging Face model ID downloads `tokenizer.json` on first use and
then uses the local Hugging Face cache. Private or gated models use the usual
`HF_TOKEN` environment variable.

## Encode text

```python
from basetenkenizer import Tokenizer

tokenizer = Tokenizer.from_model("deepseek-ai/DeepSeek-V3.2")
encoding = tokenizer.encode(
    "A very long prompt that is now much faster.",
    add_special_tokens=False,
)

print(encoding.ids)
print(tokenizer.decode(encoding.ids))
```

`Tokenizer.from_file("tokenizer.json")` loads a local tokenizer. Encoding
objects expose `ids`, `attention_mask`, `type_ids`, and
`special_tokens_mask`; selected fields can be moved into NumPy arrays with
`encoding.into_numpy(...)`.

## Kimi K2.7 Code

Load the published tokenizer directly from its model repository:

```python
from basetenkenizer import Tokenizer

tokenizer = Tokenizer.from_model("moonshotai/Kimi-K2.7-Code")
ids = tokenizer.encode("def hello():\n    return 'world'").ids
```

For a chat prompt, preserve the boundary between template control tokens and
untrusted message content with `encode_segments`:

```python
user_message = "Write a Python HTTP server."

segments = [
    ("<|im_user|>user<|im_middle|>", True),
    (user_message, False),
    ("<|im_end|>", True),
    ("<|im_assistant|>assistant<|im_middle|><think>", True),
]

encoding = tokenizer.encode_segments(segments)
input_ids = encoding.ids
```

This is the minimal Kimi K2.7 Code user/assistant shape. Applications using
system messages, tools, images, or existing assistant messages should render
the complete official model template and retain the same control-text versus
message-text boundaries.

## Kimi K3

Kimi K3 uses XTML and a Python chat renderer rather than a Jinja chat
template. Raw text can still be encoded normally:

```python
from basetenkenizer import Tokenizer

tokenizer = Tokenizer.from_model("moonshotai/Kimi-K3")
ids = tokenizer.encode("Hello from Kimi K3").ids
```

For rendered chat, pass the official Kimi K3 renderer's `EncodeSegment`
objects to `encode_segments` as `(segment.text, segment.allow_special)`. A
minimal user message followed by an assistant-generation prefix looks like
this:

```python
user_message = "Explain speculative decoding."

segments = [
    ("<|open|>", True),
    ("message", False),
    (" role", False),
    ('="', False),
    ("user", False),
    ('"', False),
    ("<|sep|>", True),
    (user_message, False),
    ("<|close|>", True),
    ("message", False),
    ("<|sep|>", True),
    ("<|end_of_msg|>", True),
    ("<|open|>", True),
    ("message", False),
    (" role", False),
    ('="', False),
    ("assistant", False),
    ('"', False),
    ("<|sep|>", True),
    ("<|open|>", True),
    ("response", False),
    ("<|sep|>", True),
]

encoding = tokenizer.encode_segments(segments)
input_ids = encoding.ids
```

Use Kimi K3's official `build_chat_segments` renderer for production
conversations involving tools, images, reasoning content, response schemas,
or multiple message types. Do not concatenate its segments before encoding:
doing so loses the special-token safety boundary.

## Why `encode_segments`?

Each segment is a `(text, allow_special)` pair:

- `allow_special=True` recognizes tokenizer control tokens emitted by a trusted
  chat renderer.
- `allow_special=False` treats control-token-looking strings in user, tool, or
  attribute content as ordinary text.

`tiktoken_safe=True` is the default. It reproduces the chunk boundaries used
by legacy tiktoken tokenizers, including on very long inputs, so token IDs stay
compatible. Set it to `False` only when exact tiktoken parity is not required
and whole-segment BPE encoding is intentional.

Post-processing, truncation, and padding are applied after all segment IDs
have been joined.

## Use with Transformers

Call `patch_transformers` before loading a tokenizer:

```python
import basetenkenizer

basetenkenizer.patch_transformers()

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")
tokens = tokenizer("Hello, world!")
```

Transformers v4 and v5 are supported. Pass
`patch_transformers(apply_chat_template=True)` to also use the native renderer
for supported render-only `apply_chat_template(..., tokenize=False)` calls;
unsupported templates automatically fall back to Transformers.

Baseten Tokenizer is focused on inference and does not implement every Hugging
Face Tokenizers training or alignment feature.

## License

MIT
