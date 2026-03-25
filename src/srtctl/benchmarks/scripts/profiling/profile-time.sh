#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Time-range profiling script (nsys-time mode)
# Workers are wrapped with "nsys profile --delay <N> --duration <M>", so all workers
# (prefill and decode) capture the exact same wall-clock window regardless of iteration speed.
# This script only needs to generate traffic for long enough to cover that window.
#
# NOTE: The orchestrator (do_sweep.py) already waits for all workers to be healthy
# before running this script, so we don't need to wait here.

model_name="${PROFILE_MODEL_NAME:-deepseek-ai/DeepSeek-R1}"
head_node="${HEAD_NODE:-127.0.0.1}"
head_port="${HEAD_PORT:-8000}"
export AIPERF_RECORD_EXPORT_BATCH_SIZE=1  # Flush profile_export.jsonl after every x requests

# Parse arguments
n_prefill=$1
n_decode=$2
prefill_gpus=$3
decode_gpus=$4
total_gpus=$5

echo "Profiling Configuration (nsys-time mode):"
echo "  Profiling mode: ${PROFILING_MODE}"
echo "  Prefill workers: ${n_prefill}"
echo "  Decode workers: ${n_decode}"
echo "  Prefill GPUs: ${prefill_gpus}"
echo "  Decode GPUs: ${decode_gpus}"
echo "  Total GPUs: ${total_gpus}"
echo "  ISL: ${PROFILE_ISL}"
echo "  OSL: ${PROFILE_OSL}"
echo "  Concurrency: ${PROFILE_CONCURRENCY}"
echo "  Benchmark duration: ${PROFILE_BENCHMARK_DURATION_SECS}s"

# Validate required parameters
if [[ -z "${PROFILE_ISL}" || -z "${PROFILE_OSL}" ]]; then
    echo "Error: PROFILE_ISL and PROFILE_OSL must be set"
    exit 1
fi
if [[ -z "${PROFILE_CONCURRENCY}" ]]; then
    echo "Error: PROFILE_CONCURRENCY must be set"
    exit 1
fi

benchmark_duration="${PROFILE_BENCHMARK_DURATION_SECS:-300}"

# Only the prefill job drives traffic through the router
if [[ "${PROFILING_MODE}" == "prefill" ]]; then
    echo ""
    echo "Generating profiling traffic for ${benchmark_duration}s..."
    echo "$(date '+%Y-%m-%d %H:%M:%S')"

    mkdir -p /logs/artifacts/nsys_profile

    if [[ "${PROFILING_BACKEND:-sglang}" == "trtllm" ]]; then
        set -x
        aiperf profile \
            --model "${model_name}" \
            --tokenizer /model/ \
            --tokenizer-trust-remote-code \
            --endpoint-type chat \
            --endpoint /v1/chat/completions \
            --streaming \
            --url "http://${head_node}:${head_port}" \
            --synthetic-input-tokens-mean "${PROFILE_ISL}" \
            --output-tokens-mean "${PROFILE_OSL}" \
            --extra-inputs "max_tokens:${PROFILE_OSL}" \
            --extra-inputs "min_tokens:${PROFILE_OSL}" \
            --extra-inputs "ignore_eos:true" \
            --concurrency "${PROFILE_CONCURRENCY}" \
            --profile-export-level raw \
            --artifact-dir /logs/artifacts/nsys_profile \
            --benchmark-duration "${benchmark_duration}" \
            --random-seed 42 \
            -H 'Authorization: Bearer NOT USED'
        set +x
    else
        set -x
        python3 -m sglang.bench_serving \
            --backend sglang \
            --model "${model_name}" \
            --host "${head_node}" --port "${head_port}" \
            --dataset-name random \
            --max-concurrency "${PROFILE_CONCURRENCY}" \
            --random-input-len "${PROFILE_ISL}" \
            --random-output-len "${PROFILE_OSL}" \
            --random-range-ratio 1 \
            --warmup-request 0 \
            --num-prompts 99999 \
            --duration "${benchmark_duration}"
        set +x
    fi
fi

exit_code=$?

echo ""
echo "$(date '+%Y-%m-%d %H:%M:%S')"
echo "Traffic generation completed with exit code ${exit_code}"

exit ${exit_code}
