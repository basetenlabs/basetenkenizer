// Copyright 2026 Baseten
// SPDX-License-Identifier: Apache-2.0

//! CPU-only BPE cache benchmark using a local tokenizer, without downloads.
//! Run with `cargo run --release --example bpe_cache_throughput -- tokenizer.json [bytes] [threads]`.
use std::{error::Error, hint::black_box, path::Path, sync::Barrier, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or("expected local tokenizer.json path")?;
    let bytes: usize = args.next().unwrap_or_else(|| "3328".into()).parse()?;
    let threads: usize = args.next().unwrap_or_else(|| "1".into()).parse()?;
    if bytes == 0 || threads == 0 {
        return Err("bytes and threads must be positive".into());
    }
    let tokenizer = basetenkenizer::Tokenizer::from_file(Path::new(&path))?;
    // One pretoken with no whitespace: measures the BPE stage, excluding Split,
    // ByteLevel, and tokenizer construction. Warm the per-thread encoding caches.
    let input: String = (0..bytes)
        .map(|i| (b'a' + ((i * 13) % 26) as u8) as char)
        .collect();
    let expected = tokenizer.model().tokenize(&input)?;
    let barrier = Barrier::new(threads);
    let iterations = 1_000_000;
    let elapsed = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let tokenizer = &tokenizer;
                let input = &input;
                let expected = &expected;
                let barrier = &barrier;
                scope.spawn(move || {
                    for _ in 0..100 {
                        assert_eq!(tokenizer.model().tokenize(input).unwrap(), *expected);
                    }
                    barrier.wait();
                    let start = Instant::now();
                    for _ in 0..iterations {
                        black_box(tokenizer.model().tokenize(black_box(input)).unwrap());
                    }
                    start.elapsed()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .max()
            .unwrap()
    });
    println!(
        "bytes={bytes} threads={threads} tokens={} requests_per_second={:.0}",
        expected.len(),
        (iterations * threads) as f64 / elapsed.as_secs_f64()
    );
    Ok(())
}
