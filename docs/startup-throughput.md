# Startup and throughput validation

Compared pre-change `792da9f`, the original candidate `148c2d8`, and the
corrected implementation. Both original steady-state regressions are removed:

- Fancy VM patterns with class intersection and lookaround keep independently
  compiled programs per parallel handle. Cloning their `Arc<Prog>` also shared
  delegate caches and caused severe contention during parallel matching.
- BPE keeps its existing DAAC prefix lookup. Hashing an entire pretoken before
  checking encoding caches penalized long cached inputs. Construction still
  shares vocabulary storage, reduces decomposition work, validates sparse IDs
  before allocation, and omits the unused prefix table.
- PCRE2 clones still share immutable compiled code with separate match-data
  pools. Its runtime handle count, chunking, and matching behavior are retained.

## Steady-state results

Throughput relative to pre-change baseline; 1.00 means parity:

| Workload | Original candidate | Corrected implementation |
| --- | ---: | ---: |
| Fancy VM, 256 KiB varied text, 1 caller | 0.18 | 1.16 |
| Fancy VM, 256 KiB varied text, 8 callers | 0.26 | 1.02 |
| BPE cached 3,328-byte pretoken, 1 caller | 0.57 | 1.00 |
| BPE cached 3,328-byte pretoken, 8 callers | 0.57 | 1.00 |

Single-caller Fancy results are noisy; the fixed paired ratios ranged from
0.993 to 1.279. This establishes recovery of the original 5.5x slowdown,
not a new throughput improvement. Independently cloned Split callers also
recovered: 1.09 with one caller and 1.02 with eight, compared with 0.20/0.26
in the original candidate. All token-span checksums matched.

Rust 1.93.1, opt-level 3 without LTO, AMD EPYC 9575F. Each workload used three
alternating fresh-process baseline/candidate pairs, with construction excluded
and matching warmup. Split used 200 ms warmup and 1 s measurement, CPUs 32-47,
Rayon 16; BPE used 128 warmup calls per caller and 700 ms measurement, CPUs 0-15.
Baseline and candidate dependencies were identical. Outputs matched across
variants; repeated and varied inputs were measured separately.

Additional isolated PCRE2 and specialized-scanner controls did not show the
severe regression. End-to-end Joey encode checks covered short, 4 KiB and
64 KiB multilingual corpus slices, repeated/varied inputs, and one/eight callers.
A longer seven-pair, 3 s check of the noisy 4 KiB varied single-caller case gave
a paired median ratio of 0.986 (range 0.799-1.086); it does not establish a
stable throughput improvement or rule out smaller regressions. These results
cover the tested paths and host, not every model or deployment.

## Construction

Five alternating fresh-process pairs with 256 visible CPUs and the local
`baselabs/joey-1` tokenizer at revision
`9efbe485efbed9598dfca5294cbbe8da00e21908`, timing `Tokenizer::from_json` after
JSON parsing:

| | Baseline | Corrected implementation |
| --- | ---: | ---: |
| Median construction | 1,149.6 ms | 976.6 ms |

This is about 15% lower construction time. The earlier 61.5% reduction belonged
to the regressing candidate and is superseded. These component timings exclude
weights, networking, CUDA, HTTP readiness, and inference.

## Reproduce the affected paths

The examples require no downloads or GPU. Copy the same example sources into
baseline and candidate checkouts to compare them, use the same locked dependency
versions/toolchain/CPU affinity, and alternate fresh-process order. Construction
is excluded from both examples. Run each comparison several times.

```sh
cargo +1.93.1 build --locked --release --no-default-features \
  --example split_throughput --example bpe_cache_throughput
RAYON_NUM_THREADS=16 taskset -c 0-15 \
  target/release/examples/split_throughput fancy 1
RAYON_NUM_THREADS=16 taskset -c 0-15 \
  target/release/examples/split_throughput fancy 8
RAYON_NUM_THREADS=16 taskset -c 0-15 \
  target/release/examples/split_throughput pcre2 8
taskset -c 0-15 target/release/examples/bpe_cache_throughput \
  /path/to/local/tokenizer.json 3328 1
taskset -c 0-15 target/release/examples/bpe_cache_throughput \
  /path/to/local/tokenizer.json 3328 8
```

The Split example defaults to intersection plus lookahead, forcing the actual
Fancy VM fallback. Intersection alone can select the plain regex wrapper and
would miss this regression. The BPE example deliberately times the model stage,
excluding Split and ByteLevel, rather than only the fused Joey encoding path.
