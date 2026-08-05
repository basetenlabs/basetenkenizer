#!/usr/bin/env python3
"""
Ablation benchmark: run dynamo_speed.py across models, datasets, input lengths,
and batch sizes.

By default runs in --tokenize-only mode (no GPU server needed).

Usage:
    python examples/dynamo_ablation.py
    python examples/dynamo_ablation.py --models mistralai/Mistral-Nemo-Instruct-2407 -n 5
    python examples/dynamo_ablation.py --dynamo -- --tp 2 --enable-piecewise-cuda-graph
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import textwrap
from datetime import datetime
from itertools import product
from pathlib import Path

# ── Defaults ──────────────────────────────────────────────────────────

DEFAULT_MODELS = [
    "deepseek-ai/DeepSeek-V3.2",
    "MiniMaxAI/MiniMax-M2.1",
    "openai/gpt-oss-120b",
    "mistralai/Mistral-Nemo-Instruct-2407",
]
DEFAULT_DATASETS = ["sharegpt", "longbench"]
DEFAULT_INPUT_LENS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 100000]
DEFAULT_BATCH_SIZES = [1, 8, 32, 128]
DEFAULT_SHARED_PREFIXES = [0.0]
DEFAULT_NUM_PROMPTS = 50
DEFAULT_WARMUP = 10


# ── Helpers ───────────────────────────────────────────────────────────


def _short_name(name: str) -> str:
    return name.replace("/", "-")


def _extract_speedup(data: dict) -> str:
    b = data.get("baseline", {})
    p = data.get("basetenkenizer", {})
    bm = b.get("mean_latency_ms")
    pm = p.get("mean_latency_ms")
    if bm and pm and pm > 0:
        return f"{bm / pm:.2f}"
    return "?"


def _fmt(v: float | None, fmt: str = ".2f") -> str:
    return f"{v:{fmt}}" if v is not None else ""


# ── CLI ───────────────────────────────────────────────────────────────


def _split_at_double_dash(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        idx = argv.index("--")
        return argv[:idx], argv[idx + 1 :]
    return argv, []


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",")]


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",")]


def _parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",")]


def main(argv: list[str] | None = None) -> None:
    raw = argv if argv is not None else sys.argv[1:]
    our_argv, server_extra = _split_at_double_dash(raw)

    parser = argparse.ArgumentParser(
        description="Run dynamo_speed.py across models, datasets, input lengths, "
        "and batch sizes. By default runs in --tokenize-only mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              %(prog)s
              %(prog)s --models mistralai/Mistral-Nemo-Instruct-2407 -n 5
              %(prog)s --dynamo -- --tp 2 --enable-piecewise-cuda-graph
        """),
    )
    parser.add_argument(
        "--models",
        type=_parse_str_list,
        default=DEFAULT_MODELS,
        help="Comma-separated model list",
    )
    parser.add_argument(
        "--datasets",
        type=_parse_str_list,
        default=DEFAULT_DATASETS,
        help="Comma-separated: sharegpt,longbench (default: both)",
    )
    parser.add_argument(
        "--input-lens",
        type=_parse_int_list,
        default=DEFAULT_INPUT_LENS,
        help="Comma-separated token lengths (default: 512,1024,2048,4096,8192,16384)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=_parse_int_list,
        default=DEFAULT_BATCH_SIZES,
        help="Comma-separated batch sizes (default: 1,8,32,128)",
    )
    parser.add_argument(
        "--shared-prefixes",
        type=_parse_float_list,
        default=DEFAULT_SHARED_PREFIXES,
        help="Comma-separated shared-prefix fractions (default: 0.0). "
             "E.g. 0.0,0.5,0.9 sweeps no prefix, 50%%, and 90%%.",
    )
    parser.add_argument(
        "-n",
        "--num-prompts",
        type=int,
        default=DEFAULT_NUM_PROMPTS,
        help=f"Prompts per run (default: {DEFAULT_NUM_PROMPTS})",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"Warmup prompts per run (default: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--dynamo",
        action="store_true",
        help="Run full Dynamo server benchmarks (not just tokenize-only)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use dynamo.mocker instead of dynamo.sglang as the backend",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: dynamo_ablation_TIMESTAMP)",
    )
    args = parser.parse_args(our_argv)

    output_dir = args.output_dir or f"dynamo_ablation_{datetime.now():%Y%m%d_%H%M%S}"
    mode = "dynamo (full server)" if args.dynamo else "tokenize-only"
    combos = list(product(args.models, args.datasets, args.input_lens, args.batch_sizes, args.shared_prefixes))
    total = len(combos)

    # ── Print configuration ───────────────────────────────────────────
    print("Dynamo Ablation Benchmark")
    print("\u2550" * 50)
    print(f"  Mode:         {mode}")
    print(f"  Models:       {' '.join(args.models)}")
    print(f"  Datasets:     {' '.join(args.datasets)}")
    print(f"  Input lens:   {' '.join(str(x) for x in args.input_lens)}")
    print(f"  Batch sizes:  {' '.join(str(x) for x in args.batch_sizes)}")
    print(f"  Shared pfx:   {' '.join(str(x) for x in args.shared_prefixes)}")
    print(f"  Num prompts:  {args.num_prompts}")
    print(f"  Warmup:       {args.warmup}")
    print(f"  Output dir:   {output_dir}")
    print(f"  Total runs:   {total}")
    if server_extra:
        print(f"  Server args:  {' '.join(server_extra)}")
    print("\u2550" * 50)
    print()

    os.makedirs(output_dir, exist_ok=True)

    # ── Run matrix ────────────────────────────────────────────────────
    failed = 0
    results: list[dict] = []

    for run_idx, (model, dataset, input_len, bs, sp) in enumerate(combos, 1):
        sp_tag = f"_sp{sp}" if sp > 0 else ""
        tag = f"{_short_name(model)}_{dataset}_len{input_len}_bs{bs}{sp_tag}"
        json_file = os.path.join(output_dir, f"{tag}.json")
        log_file = os.path.join(output_dir, f"{tag}.log")

        sp_msg = f" shared_prefix={sp}" if sp > 0 else ""
        print(f"[{run_idx}/{total}] model={model} dataset={dataset} "
              f"input_len={input_len} batch={bs}{sp_msg}")

        cmd = [
            sys.executable, "examples/dynamo_speed.py", model,
            "--dataset", dataset,
            "--num-prompts", str(args.num_prompts),
            "--warmup", str(args.warmup),
            "--input-len", str(input_len),
            "--batch-size", str(bs),
            "--output", json_file,
        ]
        if sp > 0:
            cmd.extend(["--shared-prefix", str(sp)])
        if not args.dynamo:
            cmd.append("--tokenize-only")
        if args.mock:
            cmd.append("--mock")
        if server_extra:
            cmd.append("--")
            cmd.extend(server_extra)

        print(f"  $ {' '.join(cmd)}")

        with open(log_file, "w") as log_fh:
            ret = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT)

        if ret.returncode == 0 and os.path.isfile(json_file):
            with open(json_file) as f:
                data = json.load(f)
            data["dataset"] = dataset
            data["input_len"] = input_len
            data["batch_size"] = bs
            data["shared_prefix"] = sp
            results.append(data)
            speedup = _extract_speedup(data)
            print(f"  -> done (speedup: {speedup}x)")
        else:
            failed += 1
            print(f"  -> FAILED (see {log_file})")

    # ── Combine results into JSON + CSV ───────────────────────────────
    combined_json = os.path.join(output_dir, "combined.json")
    combined_csv = os.path.join(output_dir, "combined.csv")

    results.sort(key=lambda r: (
        r.get("model", ""), r.get("dataset", ""),
        r.get("input_len", 0), r.get("batch_size", 0),
        r.get("shared_prefix", 0.0),
    ))

    with open(combined_json, "w") as f:
        json.dump(results, f, indent=2)

    csv_fields = [
        "model", "dataset", "input_len", "batch_size", "shared_prefix",
        "bl_mean_ms", "bl_median_ms", "bl_p99_ms", "bl_throughput_tok_s",
        "ft_mean_ms", "ft_median_ms", "ft_p99_ms", "ft_throughput_tok_s",
        "speedup",
    ]
    with open(combined_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for r in results:
            b = r.get("baseline", {})
            p = r.get("basetenkenizer", {})
            bm = b.get("mean_latency_ms")
            pm = p.get("mean_latency_ms")
            w.writerow({
                "model": r.get("model", ""),
                "dataset": r.get("dataset", ""),
                "input_len": r.get("input_len", ""),
                "batch_size": r.get("batch_size", ""),
                "shared_prefix": r.get("shared_prefix", 0.0),
                "bl_mean_ms": _fmt(bm),
                "bl_median_ms": _fmt(b.get("median_latency_ms")),
                "bl_p99_ms": _fmt(b.get("p99_latency_ms")),
                "bl_throughput_tok_s": _fmt(b.get("input_throughput")),
                "ft_mean_ms": _fmt(pm),
                "ft_median_ms": _fmt(p.get("median_latency_ms")),
                "ft_p99_ms": _fmt(p.get("p99_latency_ms")),
                "ft_throughput_tok_s": _fmt(p.get("input_throughput")),
                "speedup": f"{bm / pm:.2f}" if bm and pm and pm > 0 else "",
            })

    # ── Summary ───────────────────────────────────────────────────────
    print()
    print("\u2550" * 66)
    print(f"  Ablation complete: {len(combos)} runs, {failed} failed")
    print(f"  Combined JSON: {combined_json}")
    print(f"  Combined CSV:  {combined_csv}")
    print(f"  Per-run logs:  {output_dir}/*.log")
    print("\u2550" * 66)

    # ── Summary table ─────────────────────────────────────────────────
    if not results:
        print("\n  No results to display.")
        sys.exit(1 if failed else 0)

    print()
    print(
        f"{'MODEL':<40} {'DATASET':<12} {'INPUT_LEN':>10} {'BATCH':>6} {'PREFIX':>7} "
        f"{'BL mean(ms)':>12} {'FT mean(ms)':>12} "
        f"{'BL tok/s':>10} {'FT tok/s':>10} {'SPEEDUP':>8}"
    )
    print("\u2500" * 133)

    for r in results:
        model = r.get("model", "?")
        short = model.rsplit("/", 1)[-1] if "/" in model else model
        if len(short) > 38:
            short = short[:35] + "..."

        b = r.get("baseline", {})
        p = r.get("basetenkenizer", {})
        bm = b.get("mean_latency_ms")
        pm = p.get("mean_latency_ms")
        bt = b.get("input_throughput")
        pt = p.get("input_throughput")

        sp = r.get("shared_prefix", 0.0)
        sp_s = f"{sp:.0%}" if sp > 0 else "-"

        bm_s = f"{bm:.1f}" if bm is not None else "-"
        pm_s = f"{pm:.1f}" if pm is not None else "-"
        bt_s = f"{bt:.0f}" if bt is not None else "-"
        pt_s = f"{pt:.0f}" if pt is not None else "-"
        speedup = f"{bm / pm:.2f}x" if bm and pm and pm > 0 else "-"

        print(
            f"{short:<40} {r.get('dataset', '?'):<12} "
            f"{str(r.get('input_len', '?')):>10} {str(r.get('batch_size', '?')):>6} {sp_s:>7} "
            f"{bm_s:>12} {pm_s:>12} {bt_s:>10} {pt_s:>10} {speedup:>8}"
        )

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
