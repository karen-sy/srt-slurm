#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test that --num-dataset-entries 1 makes all aiperf requests use the same prompt.

This validates the single_prompt mode for decode-only benchmarking:
- With num-dataset-entries=1: All requests use identical prompts (1 prefill warms cache for all)
- With num-dataset-entries=N: N unique prompts (need N prefills to warm cache)

Usage:
    python test_single_prompt.py
    python test_single_prompt.py --num-requests 10
"""

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


class MockLLMHandler(BaseHTTPRequestHandler):
    """Mock LLM server that logs received prompts."""

    prompts_received: list = []
    lock = threading.Lock()

    def log_message(self, *args):
        pass

    def do_GET(self):
        if "/models" in self.path:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": "test-model"}]}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        data = json.loads(body)
        prompt = data.get("messages", [{}])[-1].get("content", "")
        h = hashlib.md5(prompt.encode()).hexdigest()[:8]

        with self.lock:
            self.prompts_received.append(h)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(3):
            self.wfile.write(f'data: {{"choices":[{{"delta":{{"content":"tok{i} "}}}}]}}\n\n'.encode())
        self.wfile.write(b"data: [DONE]\n\n")


def run_server(port: int):
    server = HTTPServer(("127.0.0.1", port), MockLLMHandler)
    server.serve_forever()


def run_aiperf_test(port: int, num_requests: int, num_dataset_entries: int, seed: int) -> tuple[int, list[str]]:
    """Run aiperf and return (return_code, list of prompt hashes received)."""
    MockLLMHandler.prompts_received = []

    result = subprocess.run(
        [
            "aiperf",
            "profile",
            "--url",
            f"http://127.0.0.1:{port}",
            "--model",
            "test-model",
            "--tokenizer",
            "gpt2",
            "--endpoint-type",
            "chat",
            "--endpoint",
            "/v1/chat/completions",
            "--streaming",
            "--synthetic-input-tokens-mean",
            "50",
            "--output-tokens-mean",
            "3",
            "--num-dataset-entries",
            str(num_dataset_entries),
            "--request-count",
            str(num_requests),
            "--concurrency",
            str(num_requests),
            "--warmup-request-count",
            "1",
            "--random-seed",
            str(seed),
            "--artifact-dir",
            f"/tmp/aiperf_single_prompt_test_{num_dataset_entries}",
            "--ui",
            "simple",
        ],
        capture_output=True,
        timeout=120,
    )

    return result.returncode, MockLLMHandler.prompts_received.copy()


def main():
    parser = argparse.ArgumentParser(description="Test single_prompt mode for decode-only benchmark")
    parser.add_argument("--port", type=int, default=18899, help="Port for mock server")
    parser.add_argument("--num-requests", type=int, default=5, help="Number of requests to send")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 60)
    print("Testing single_prompt mode (--num-dataset-entries behavior)")
    print("=" * 60)
    print()

    # Start mock server
    print(f"Starting mock server on port {args.port}...")
    server_thread = threading.Thread(target=run_server, args=(args.port,), daemon=True)
    server_thread.start()
    time.sleep(0.5)

    # Test 1: num-dataset-entries=1 (single prompt mode)
    print()
    print("-" * 60)
    print(f"TEST 1: num-dataset-entries=1, requests={args.num_requests}")
    print("-" * 60)

    rc1, hashes1 = run_aiperf_test(args.port, args.num_requests, num_dataset_entries=1, seed=args.seed)
    unique1 = len(set(hashes1))

    print(f"  Return code: {rc1}")
    print(f"  Requests received: {len(hashes1)}")
    print(f"  Unique prompts: {unique1}")
    print(f"  Hashes: {hashes1}")

    # Test 2: num-dataset-entries=N (multiple prompts)
    print()
    print("-" * 60)
    print(f"TEST 2: num-dataset-entries={args.num_requests}, requests={args.num_requests}")
    print("-" * 60)

    rc2, hashes2 = run_aiperf_test(args.port, args.num_requests, num_dataset_entries=args.num_requests, seed=args.seed)
    unique2 = len(set(hashes2))

    print(f"  Return code: {rc2}")
    print(f"  Requests received: {len(hashes2)}")
    print(f"  Unique prompts: {unique2}")
    print(f"  Hashes: {hashes2}")

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    print(f"  num-dataset-entries=1: {unique1} unique prompt(s)")
    print(f"  num-dataset-entries={args.num_requests}: {unique2} unique prompt(s)")
    print()

    success = True
    if unique1 == 1:
        print("  ✓ single_prompt mode works: all requests use SAME prompt")
    else:
        print("  ✗ FAIL: single_prompt mode should produce 1 unique prompt")
        success = False

    if unique2 > 1:
        print(f"  ✓ multi-prompt mode works: {unique2} different prompts")
    else:
        print("  ✗ FAIL: multi-prompt mode should produce multiple unique prompts")
        success = False

    print()
    print("=" * 60)

    if success:
        print("All tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
