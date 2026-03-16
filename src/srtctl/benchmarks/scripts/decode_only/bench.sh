#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Decode-Only Benchmark: Measures pure decode throughput via KV cache hits
#
# Follows Dynamo profiler pattern:
#   - Warmup: Send N identical prompts to populate KV cache
#   - Measurement: Send same N prompts (same seed) → cache hits → skip prefill
#
# Key: Single-turn requests with same seed = identical prompts = cache hits
#
# REQUIRES: enable_block_reuse: true in TRT-LLM KV cache config

set -e

export PYTHONUNBUFFERED=1
export AIPERF_RECORD_EXPORT_BATCH_SIZE=3  # Flush profile_export.jsonl after every 3 requests

ENDPOINT=$1
ISL=$2
OSL=$3
MODEL_PATH=${4:-/model/}
MODEL_NAME=${5:-"model"}
SEED=${6:-42}
CONCURRENCIES_STR=${7:-"1x10x25"}

MODEL_PATH="/model/"

SERVER_METRICS_ARGS=()
if [ -n "${AIPERF_SERVER_METRICS_URLS:-}" ]; then
    IFS=',' read -r -a server_metrics_urls <<< "${AIPERF_SERVER_METRICS_URLS}"
    if [ ${#server_metrics_urls[@]} -gt 0 ]; then
        SERVER_METRICS_ARGS+=(--server-metrics "${server_metrics_urls[@]}")
    fi
fi

SCRIPT_DIR="$(dirname "$0")"

wait_for_model_ready() {
    echo "Waiting for model '${MODEL_NAME}' at ${ENDPOINT}/v1/models..."
    while ! curl -s "${ENDPOINT}/v1/models" | jq -e --arg model "$MODEL_NAME" '.data[]? | select(.id == $model)' >/dev/null 2>&1; do
        echo "[$(date '+%H:%M:%S')] Model not ready, retrying in 5s..."
        sleep 5
    done
    echo "Model '${MODEL_NAME}' is available!"
    curl -s "${ENDPOINT}/v1/models" | jq .
}

run_aiperf() {
    local phase=$1
    local artifact_dir=$2
    local seed=$3
    local num_requests=$4
    local concurrency=$5
    local warmup_count=${6:-0}  # 0 = no internal warmup (omit flag)
    local osl_override=${7:-$OSL}  # Optional OSL override, defaults to global OSL
    
    echo ""
    echo "[${phase}] aiperf: requests=${num_requests}, concurrency=${concurrency}, seed=${seed}, warmup=${warmup_count}, osl=${osl_override}"
    echo "Artifact dir: ${artifact_dir}"
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    
    mkdir -p "$artifact_dir"
    
    # Build warmup args (omit if 0 to skip aiperf's internal warmup phase)
    local warmup_args=()
    if [ "$warmup_count" -gt 0 ]; then
        warmup_args=("--warmup-request-count" "$warmup_count")
    fi
    
    # Single-turn requests with fixed ISL/OSL (matches Dynamo profiler pattern)
    # Same seed + same num-dataset-entries = identical prompts for cache hits
    aiperf profile --artifact-dir "$artifact_dir" \
        --model "$MODEL_NAME" \
        --tokenizer "$MODEL_PATH" \
        --tokenizer-trust-remote-code \
        --endpoint-type chat \
        --endpoint /v1/chat/completions \
        --streaming \
        --url "$ENDPOINT" \
        --random-seed 42 \
        --extra-inputs "ignore_eos:true" \
        --extra-inputs '{"nvext":{"ignore_eos":true}}' \
        --extra-inputs "max_tokens:${osl_override}" \
        --extra-inputs "min_tokens:${osl_override}" \
        --synthetic-input-tokens-mean "$ISL" \
        --synthetic-input-tokens-stddev 0 \
        --output-tokens-mean "$osl_override" \
        --output-tokens-stddev 0 \
        --num-dataset-entries 1 \
        --concurrency "$concurrency" \
        --request-count "$num_requests" \
        "${warmup_args[@]}" \
        --workers-max 200 \
        --request-timeout-seconds 1800 \
        --profile-export-level records \
        -H 'Authorization: Bearer NOT USED' \
        -H 'Accept: text/event-stream' \
        --record-processors 8 \
        "${SERVER_METRICS_ARGS[@]}" \
        --ui simple
    
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[${phase}] Complete"
}

ulimit -n 600000 2>/dev/null || ulimit -n 65536 2>/dev/null || true

wait_for_model_ready

EPOCH=$(date +%s)

echo "=============================================="
echo "Decode-Only Benchmark"
echo "=============================================="
echo "Endpoint: ${ENDPOINT}"
echo "ISL: ${ISL}, OSL: ${OSL}"
echo "Concurrencies: ${CONCURRENCIES_STR}"
echo "Random Seed: ${SEED}"
echo "Model: ${MODEL_NAME}"
echo "=============================================="

result_dir="/logs/decode_only_${EPOCH}"
mkdir -p "$result_dir"

# Parse concurrencies
IFS='x' read -r -a CONCURRENCY_LIST <<< "$CONCURRENCIES_STR"

cat > "${result_dir}/input_config.json" <<EOF
{
    "benchmark_type": "decode_only",
    "isl": ${ISL},
    "osl": ${OSL},
    "concurrencies": [$(IFS=,; echo "${CONCURRENCY_LIST[*]}")],
    "seed": ${SEED},
    "endpoint": "${ENDPOINT}",
    "model": "${MODEL_NAME}"
}
EOF

# Summary file
echo "concurrency,warmup_ttft_ms,warmup_itl_ms,measurement_ttft_ms,measurement_itl_ms,measurement_throughput_tok_s" > "${result_dir}/summary.csv"

echo ""
echo "PHASE 1: WARMUP (pre-populate KV cache, OSL=1)"
echo "--------------------------------------------------------------"
# Warmup: send requests to populate KV cache (1 internal warmup for JIT)
# OSL=1 for warmup to minimize decode time while still populating cache
# Run once before all concurrency tests
run_aiperf "WARMUP" "${result_dir}/warmup" "$SEED" 120 5 1 1

for conc in "${CONCURRENCY_LIST[@]}"; do
    echo ""
    echo "############################################################"
    echo "# CONCURRENCY: ${conc}"
    echo "############################################################"
    
    # Calculate num_requests = min(concurrency * 2, 64)
    # Fewer requests = faster benchmark, cache behavior still works with same seed
    num_requests=$((conc * 2))
    if [ "$num_requests" -gt 128 ]; then
        num_requests=128
    fi
    
    conc_dir="${result_dir}/concurrency_${conc}"
    mkdir -p "$conc_dir"
    
    echo ""
    echo "PHASE 2: MEASUREMENT (${num_requests} requests at concurrency ${conc}, expect cache hits)"
    echo "--------------------------------------------------------------"
    # Measurement: same seed = same prompts, NO internal warmup to avoid cache eviction
    # The 0 means aiperf sends requests immediately without its own warmup phase
    run_aiperf "MEASUREMENT" "${conc_dir}/measurement" "$SEED" "$num_requests" "$conc" 0
    
    # Extract metrics
    warmup_json="${conc_dir}/warmup/profile_export_aiperf.json"
    measurement_json="${conc_dir}/measurement/profile_export_aiperf.json"
    
    if [ -f "$warmup_json" ] && [ -f "$measurement_json" ]; then
        warmup_ttft=$(python3 -c "import json; d=json.load(open('$warmup_json')); print(f\"{d.get('time_to_first_token',{}).get('avg',0):.2f}\")" 2>/dev/null || echo "N/A")
        warmup_itl=$(python3 -c "import json; d=json.load(open('$warmup_json')); print(f\"{d.get('inter_token_latency',{}).get('avg',0):.2f}\")" 2>/dev/null || echo "N/A")
        meas_ttft=$(python3 -c "import json; d=json.load(open('$measurement_json')); print(f\"{d.get('time_to_first_token',{}).get('avg',0):.2f}\")" 2>/dev/null || echo "N/A")
        meas_itl=$(python3 -c "import json; d=json.load(open('$measurement_json')); print(f\"{d.get('inter_token_latency',{}).get('avg',0):.2f}\")" 2>/dev/null || echo "N/A")
        meas_thpt=$(python3 -c "import json; d=json.load(open('$measurement_json')); print(f\"{d.get('output_token_throughput',{}).get('avg',0):.2f}\")" 2>/dev/null || echo "N/A")
        
        echo "${conc},${warmup_ttft},${warmup_itl},${meas_ttft},${meas_itl},${meas_thpt}" >> "${result_dir}/summary.csv"
        
        echo ""
        echo "--- Concurrency ${conc} Results ---"
        echo "Warmup:      TTFT=${warmup_ttft}ms, ITL=${warmup_itl}ms"
        echo "Measurement: TTFT=${meas_ttft}ms, ITL=${meas_itl}ms, Throughput=${meas_thpt} tok/s"
    fi
done

echo ""
echo "=============================================="
echo "SUMMARY"
echo "=============================================="
echo ""
column -t -s, "${result_dir}/summary.csv"
echo ""
echo "Results in: $result_dir"
echo "=============================================="

ls -la "$result_dir"
