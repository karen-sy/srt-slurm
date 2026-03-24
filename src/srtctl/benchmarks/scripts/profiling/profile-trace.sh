#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Profiling script using trace-replay traffic (nsys-trace mode)
# Sends /start_profile API calls to workers and generates traffic via aiperf trace replay.
#
# NOTE: The orchestrator (do_sweep.py) already waits for all workers to be healthy
# before running this script, so we don't need to wait here.

model_name="${PROFILE_MODEL_NAME:-deepseek-ai/DeepSeek-R1}"
head_node="${HEAD_NODE:-127.0.0.1}"
head_port="${HEAD_PORT:-8000}"
export AIPERF_RECORD_EXPORT_BATCH_SIZE=1  # Flush profile_export.jsonl after every request


# Parse arguments
n_prefill=$1
n_decode=$2
prefill_gpus=$3
decode_gpus=$4
total_gpus=$5

echo "Profiling Configuration:"
echo "  Profiling mode: ${PROFILING_MODE}"
echo "  Profiling dir: ${SGLANG_TORCH_PROFILER_DIR}"
echo "  Prefill workers: ${n_prefill}"
echo "  Decode workers: ${n_decode}"
echo "  Prefill GPUs: ${prefill_gpus}"
echo "  Decode GPUs: ${decode_gpus}"
echo "  Total GPUs: ${total_gpus}"
echo "  Trace file: ${PROFILE_TRACE_FILE}"
echo "  Concurrency: ${PROFILE_CONCURRENCY}"

# Validate required parameters
if [[ -z "${PROFILE_TRACE_FILE}" ]]; then
    echo "Error: PROFILE_TRACE_FILE must be set"
    exit 1
fi
if [[ -z "${PROFILE_CONCURRENCY}" ]]; then
    echo "Error: PROFILE_CONCURRENCY must be set"
    exit 1
fi

# Parse leader IP lists from environment (comma-separated)
IFS=',' read -r -a PREFILL_IPS <<< "${PROFILE_PREFILL_IPS:-}"
IFS=',' read -r -a DECODE_IPS <<< "${PROFILE_DECODE_IPS:-}"
IFS=',' read -r -a AGG_IPS <<< "${PROFILE_AGG_IPS:-}"

# Get phase-specific start/stop steps
get_phase_start_step() {
    local phase="$1"
    local var_name="PROFILE_${phase}_START_STEP"
    echo "${!var_name:-0}"
}

get_phase_stop_step() {
    local phase="$1"
    local var_name="PROFILE_${phase}_STOP_STEP"
    echo "${!var_name:-50}"
}

# Start profiling on a worker
start_profile_on_worker() {
    local ip="$1"
    local start_step="$2"
    local stop_step="$3"

    if [[ -z "${ip}" ]]; then
        return
    fi

    local num_steps=$((stop_step - start_step))
    if [[ "${num_steps}" -le 0 ]]; then
        echo "Error: invalid step range: start=${start_step} stop=${stop_step}"
        return 1
    fi

    # Determine activities based on profiler type
    local ACTIVITIES
    if [[ -n "${SGLANG_TORCH_PROFILER_DIR}" ]]; then
        ACTIVITIES='["CPU", "GPU", "MEM"]'
    else
        ACTIVITIES='["CUDA_PROFILER"]'
    fi

    echo "Starting profiling on http://${ip}:30000 (steps ${start_step}-${stop_step})"
    curl -sS -X POST "http://${ip}:30000/start_profile" \
        -H "Content-Type: application/json" \
        -d "{\"start_step\": ${start_step}, \"num_steps\": ${num_steps}, \"activities\": ${ACTIVITIES}}" || true
}

# Check if we have any workers to profile
if [[ "${#PREFILL_IPS[@]}" -eq 0 && "${#DECODE_IPS[@]}" -eq 0 && "${#AGG_IPS[@]}" -eq 0 ]]; then
    echo "Error: No worker IPs provided for profiling"
    echo "Set PROFILE_PREFILL_IPS, PROFILE_DECODE_IPS, or PROFILE_AGG_IPS"
    exit 1
fi

# Create profiling output directory
if [[ -n "${SGLANG_TORCH_PROFILER_DIR}" ]]; then
    mkdir -p "${SGLANG_TORCH_PROFILER_DIR}" 2>/dev/null || true
fi

echo ""
echo "Starting profiling..."
echo "$(date '+%Y-%m-%d %H:%M:%S')"

set -x

# Get phase-specific steps
prefill_start=$(get_phase_start_step PREFILL)
prefill_stop=$(get_phase_stop_step PREFILL)
decode_start=$(get_phase_start_step DECODE)
decode_stop=$(get_phase_stop_step DECODE)
agg_start=$(get_phase_start_step AGG)
agg_stop=$(get_phase_stop_step AGG)

# Start profiling on all workers
for ip in "${PREFILL_IPS[@]}"; do
    start_profile_on_worker "${ip}" "${prefill_start}" "${prefill_stop}"
done
for ip in "${DECODE_IPS[@]}"; do
    start_profile_on_worker "${ip}" "${decode_start}" "${decode_stop}"
done
for ip in "${AGG_IPS[@]}"; do
    start_profile_on_worker "${ip}" "${agg_start}" "${agg_stop}"
done

# Only the prefill profiling job needs to generate traffic through the router.
if [[ "${PROFILING_MODE}" == "prefill" ]]; then
    echo ""
    echo "Generating profiling traffic from trace file..."
    echo "Trace file: ${PROFILE_TRACE_FILE}"

    # Install aiperf
    pip install "aiperf @ git+https://github.com/ai-dynamo/aiperf.git@b1dd72f2a1ca58b6e72bbaba66c1d76114b856a0"
    pip install tiktoken

    mkdir -p /logs/artifacts/nsys_profile

    # Increase file descriptor limit for high concurrency
    ulimit -n 600000 2>/dev/null || ulimit -n 65536 2>/dev/null || true

    export AIPERF_HTTP_SO_RCVTIMEO=120

    aiperf profile \
        -m "${model_name}" \
        --tokenizer "${model_name}" \
        --tokenizer-trust-remote-code \
        --url "http://${head_node}:${head_port}" \
        --streaming \
        --input-file "${PROFILE_TRACE_FILE}" \
        --custom-dataset-type mooncake_trace \
        --prompt-corpus coding \
        --concurrency "${PROFILE_CONCURRENCY}" \
        --benchmark-duration 300 \
        --benchmark-grace-period 60 \
        --workers-max 200 \
        --request-timeout-seconds 1200 \
        --record-processors 8 \
        --profile-export-level raw \
        --export-http-trace \
        --ui dashboard \
        --artifact-dir /logs/artifacts/nsys_concurrency_${PROFILE_CONCURRENCY}
fi

exit_code=$?
set +x

echo ""
echo "$(date '+%Y-%m-%d %H:%M:%S')"
echo "Profiling completed with exit code ${exit_code}"
if [[ -n "${SGLANG_TORCH_PROFILER_DIR}" ]]; then
    echo "Profiling results saved to ${SGLANG_TORCH_PROFILER_DIR}"
fi

exit ${exit_code}
