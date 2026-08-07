#!/usr/bin/env python3
# Copyright 2026 Baseten
# SPDX-License-Identifier: Apache-2.0

"""Enforce the repository's Apache/proprietary copyright boundary."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]

BASETEN_APACHE = {
    "docs/perf/specialized-split-scanners.md",
    "python/basetenkenizer/tiktoken.py",
    "scripts/check_copyright_headers.py",
    "scripts/generate_cargo_license_inventory.py",
    "src/chat_template/mod.rs",
    "src/chat_template/runtime.rs",
    "src/chat_template/special_tokens.rs",
    "tests/fixtures/PROVENANCE.md",
}

BASETEN_PROPRIETARY = {
    "examples/long_context_bench.rs",
    "examples/serving_split_bench.rs",
    "examples/tiktoken_long_context.py",
    "python/tests/test_async_stub.py",
    "python/tests/test_chat_template.py",
    "python/tests/test_split_special_tokens.py",
    "python/tests/test_tiktoken_conversion.py",
    "python/tests/test_unsupported_encode_options.py",
}

# These files originated in fastokens. They must never be swept into Baseten's
# proprietary allowlist, even when Baseten has subsequently modified them.
FASTOKENS_DERIVED = {
    "examples/ablation.sh",
    "examples/dynamo_ablation.py",
    "examples/dynamo_speed.py",
    "examples/get_step_kinds.rs",
    "examples/print_pipeline.rs",
    "examples/profile_sample.rs",
    "examples/profile_stages.rs",
    "examples/sglang_quality.py",
    "examples/sglang_speed.py",
    "examples/simple_bench.rs",
    "examples/validate_model.sh",
    "python/basetenkenizer/__init__.py",
    "python/basetenkenizer/_compat.py",
    "python/basetenkenizer/_native.pyi",
    "python/src/lib.rs",
    "python/tests/test_patch_transformers.py",
    "python/tests/test_thread_safety.py",
    "src/added_tokens.rs",
    "src/decoders.rs",
    "src/decoders/byte_level.rs",
    "src/json_structs.rs",
    "src/lib.rs",
    "src/models.rs",
    "src/models/bpe.rs",
    "src/normalizers.rs",
    "src/normalizers/nfc.rs",
    "src/post_processors.rs",
    "src/pre_tokenized.rs",
    "src/pre_tokenizers.rs",
    "src/pre_tokenizers/byte_level.rs",
    "src/pre_tokenizers/split.rs",
}

COPYRIGHT = "Copyright 2026 Baseten"
APACHE_SPDX = "SPDX-License-Identifier: Apache-2.0"
ALL_RIGHTS_RESERVED = "All rights" + " reserved."
PROPRIETARY_MARKER = "This file is proprietary" + " to Baseten"
NOT_APACHE = "not licensed under the\n"


def read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def main() -> int:
    errors: list[str] = []

    for relative_path in sorted(BASETEN_APACHE):
        text = read(relative_path)
        if COPYRIGHT not in text or APACHE_SPDX not in text:
            errors.append(f"{relative_path}: missing Baseten Apache-2.0 header")
        if ALL_RIGHTS_RESERVED in text:
            errors.append(f"{relative_path}: Apache file marked all rights reserved")

    for relative_path in sorted(BASETEN_PROPRIETARY):
        text = read(relative_path)
        if COPYRIGHT not in text or ALL_RIGHTS_RESERVED not in text or NOT_APACHE not in text:
            errors.append(f"{relative_path}: missing Baseten proprietary header")

    grep = subprocess.run(
        ["git", "grep", "-l", "-F", PROPRIETARY_MARKER],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if grep.returncode not in (0, 1):
        errors.append(f"git grep failed: {grep.stderr.strip()}")
    actual = set(grep.stdout.splitlines())
    expected = BASETEN_PROPRIETARY
    if actual != expected:
        unexpected = sorted(actual - expected)
        missing = sorted(expected - actual)
        if unexpected:
            errors.append(f"unexpected Baseten-proprietary files: {', '.join(unexpected)}")
        if missing:
            errors.append(f"missing Baseten-proprietary files: {', '.join(missing)}")

    wrongly_proprietary = FASTOKENS_DERIVED & actual
    if wrongly_proprietary:
        errors.append(
            "fastokens-derived files marked proprietary: "
            + ", ".join(sorted(wrongly_proprietary))
        )

    if errors:
        print("Copyright boundary check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Copyright boundary check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
