#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Validate KV cache reuse is working correctly.

Sends identical requests and compares TTFT:
- Request 1: Cache miss (real prefill) - expect high TTFT
- Request 2: Cache hit (skip prefill) - expect much lower TTFT

Usage:
    python validate_kv_reuse.py http://localhost:8000 --model model_name
    python validate_kv_reuse.py http://localhost:8000 --isl 3000 --osl 100
"""

import argparse
import json
import time
import requests


def generate_prompt(num_tokens: int, seed: int = 42) -> str:
    """Generate a prompt with approximately num_tokens tokens."""
    import random
    random.seed(seed)
    
    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "hello", "world", "python", "code", "test", "data", "model", "cache",
        "memory", "compute", "tensor", "matrix", "vector", "neural", "network",
        "layer", "attention", "transformer", "token", "embedding", "weight",
    ]
    
    # Roughly 1.3 tokens per word on average
    num_words = int(num_tokens / 1.3)
    prompt_words = [random.choice(words) for _ in range(num_words)]
    return " ".join(prompt_words)


def send_request(endpoint: str, model: str, prompt: str, max_tokens: int) -> dict:
    """Send a chat completion request and measure TTFT."""
    url = f"{endpoint}/v1/chat/completions"
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "stream": True,
        "ignore_eos": True,
        "nvext": {"ignore_eos": True},
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    
    start_time = time.perf_counter()
    first_token_time = None
    total_tokens = 0
    
    try:
        with requests.post(url, json=payload, headers=headers, stream=True, timeout=300) as response:
            response.raise_for_status()
            
            for line in response.iter_lines():
                if not line:
                    continue
                    
                line = line.decode("utf-8")
                if not line.startswith("data: "):
                    continue
                    
                data = line[6:]  # Remove "data: " prefix
                if data == "[DONE]":
                    break
                
                try:
                    chunk = json.loads(data)
                    if chunk.get("choices", [{}])[0].get("delta", {}).get("content"):
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        total_tokens += 1
                except json.JSONDecodeError:
                    continue
                    
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
    
    end_time = time.perf_counter()
    
    ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else None
    total_time_ms = (end_time - start_time) * 1000
    
    return {
        "ttft_ms": ttft_ms,
        "total_time_ms": total_time_ms,
        "tokens_generated": total_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate KV cache reuse")
    parser.add_argument("endpoint", help="Server endpoint (e.g., http://localhost:8000)")
    parser.add_argument("--model", default="model", help="Model name")
    parser.add_argument("--isl", type=int, default=1000, help="Input sequence length (tokens)")
    parser.add_argument("--osl", type=int, default=50, help="Output sequence length (tokens)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for prompt generation")
    parser.add_argument("--num-repeats", type=int, default=3, help="Number of cache-hit requests")
    args = parser.parse_args()
    
    print("=" * 60)
    print("KV Cache Reuse Validation")
    print("=" * 60)
    print(f"Endpoint: {args.endpoint}")
    print(f"Model: {args.model}")
    print(f"ISL: {args.isl}, OSL: {args.osl}")
    print(f"Seed: {args.seed}")
    print()
    
    # Generate the prompt
    prompt = generate_prompt(args.isl, args.seed)
    print(f"Generated prompt with ~{args.isl} tokens")
    print(f"Prompt preview: {prompt[:100]}...")
    print()
    
    # Request 1: Cache miss (first time seeing this prompt)
    print("-" * 60)
    print("REQUEST 1: Cache MISS (real prefill expected)")
    print("-" * 60)
    result1 = send_request(args.endpoint, args.model, prompt, args.osl)
    
    if "error" in result1:
        print(f"ERROR: {result1['error']}")
        return 1
    
    print(f"  TTFT: {result1['ttft_ms']:.2f} ms")
    print(f"  Total time: {result1['total_time_ms']:.2f} ms")
    print(f"  Tokens generated: {result1['tokens_generated']}")
    print()
    
    # Requests 2+: Cache hit (should reuse KV cache)
    cache_hit_ttfts = []
    for i in range(args.num_repeats):
        print("-" * 60)
        print(f"REQUEST {i+2}: Cache HIT expected (same prompt)")
        print("-" * 60)
        
        result = send_request(args.endpoint, args.model, prompt, args.osl)
        
        if "error" in result:
            print(f"ERROR: {result['error']}")
            continue
        
        print(f"  TTFT: {result['ttft_ms']:.2f} ms")
        print(f"  Total time: {result['total_time_ms']:.2f} ms")
        print(f"  Tokens generated: {result['tokens_generated']}")
        cache_hit_ttfts.append(result['ttft_ms'])
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if cache_hit_ttfts:
        avg_cache_hit_ttft = sum(cache_hit_ttfts) / len(cache_hit_ttfts)
        speedup = result1['ttft_ms'] / avg_cache_hit_ttft if avg_cache_hit_ttft > 0 else 0
        
        print(f"Cache MISS TTFT:     {result1['ttft_ms']:>10.2f} ms")
        print(f"Cache HIT TTFT avg:  {avg_cache_hit_ttft:>10.2f} ms")
        print(f"Speedup:             {speedup:>10.1f}x")
        print()
        
        if speedup > 2:
            print("✓ KV cache reuse is WORKING!")
            print("  TTFT dropped significantly on repeated requests.")
        elif speedup > 1.2:
            print("? KV cache reuse MAY be working.")
            print("  TTFT dropped but not as much as expected.")
            print("  Check that enable_block_reuse: true is set.")
        else:
            print("✗ KV cache reuse does NOT appear to be working.")
            print("  TTFT is similar for cache miss and hit.")
            print("  Verify enable_block_reuse: true in kv_cache_config.")
    else:
        print("Could not measure cache hit performance.")
    
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
