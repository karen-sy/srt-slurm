# Decode-Only Benchmark Design

## Goal

Measure pure decode throughput by using KV cache hits to skip prefill.

## Approach (matches Dynamo profiler)

Uses single-turn requests with identical prompts across warmup/measurement:

1. **Warmup Phase**: Send N identical prompts to populate KV cache
2. **Measurement Phase**: Send same N prompts again (same seed) → cache hits → near-zero TTFT

## Key Parameters

```bash
--num-dataset-entries N      # Limits unique prompts to N
--concurrency N              # Send all N concurrently  
--request-count N            # Total N requests
--random-seed SAME           # Same seed = same prompts
```

No multi-turn options - single-turn only for exact prompt matching.

## Requirements

- `enable_block_reuse: true` in TRT-LLM KV cache config
- Same random seed for warmup and measurement phases
- Sufficient KV cache to hold all N prompts

## What Gets Measured

| Metric | Warmup | Measurement |
|--------|--------|-------------|
| TTFT | Full prefill | Near-zero (cache hits) |
| ITL | Decode latency | Pure decode latency |
| Throughput | Mixed | Decode-only throughput |

## Default Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| ISL | 4000 | Input sequence length (tokens) |
| OSL | 600 | Output sequence length (tokens) |
| Concurrencies | 1x10x25 | Sweep across concurrency levels |

## Requirements

- `enable_block_reuse: true` in KV cache config (required for cache hits)
- Sufficient GPU memory to hold conversation histories
- Same random seed for both phases (ensures identical conversations)

## Default Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--shared-system-prompt-length` | 32000 | Shared system prompt (cached once) |
| `--user-context-prompt-length` | 3000 | Per-conversation user context |
| `--conversation-num` | 100 | Number of concurrent conversations |
| `--session-turns-mean` | 10 | Average turns per conversation |
| `--session-turns-stddev` | 3 | Variation in turns |
| `--benchmark-duration` | 900 | Max duration per phase (seconds) |
| `--concurrency` | varies | Concurrent requests (default sweep: 1x10x25) |

Requests per concurrency: `min(concurrency * 10, 50)`

## KV Cache Sizing

### Per-Conversation Token Budget

```
Shared system prompt:  32,000 tokens (cached once, reused across all conversations)
User context:           3,000 tokens per conversation
Per turn:               ISL + OSL tokens
```

### Example: ISL=500, OSL=500

```
Per conversation: 3,000 + (10 turns × 1,000) = 13,000 tokens
100 conversations: 1,300,000 unique tokens
Shared prompt:    32,000 tokens (amortized)
Total:            ~1.33M tokens
```

### Example: ISL=4000, OSL=600

```
Per conversation: 3,000 + (10 turns × 4,600) = 49,000 tokens
100 conversations: 4,900,000 unique tokens
Shared prompt:    32,000 tokens (amortized)
Total:            ~4.93M tokens
```

### Memory Estimate (FP8 KV Cache)

For large MoE models (~600B params), rough KV cache sizing:
- ~0.5-1 KB per token
- 4.93M tokens × 1 KB ≈ **5 GB KV cache**

### Hardware Fit (8x GB200 + Kimi-K2.5 NVFP4)

```
Total HBM:           8 × 192 GB = 1,536 GB
Model weights:       ~500 GB (Kimi-K2.5 NVFP4)
Remaining:           1,036 GB
KV cache allocation: 70% of remaining = 725 GB
```

Benchmark KV requirement: **~5 GB**
Available KV cache: **725 GB**

Fits with ~145x headroom. No memory pressure concerns.
