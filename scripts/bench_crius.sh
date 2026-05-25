#!/bin/bash

set -euo pipefail

if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <num_mlus> <batch_size> <seq_len> <tp_degree> <pp_degree> <dp_degree>"
    exit 1
fi

NUM_MLUS=$1
BATCH_SIZE=$2
SEQ_LEN=$3
TP_DEGREE=$4
PP_DEGREE=$5
DP_DEGREE=$6
TUNER_ENV_FILE=$(mktemp "${TMPDIR:-/tmp}/crius_parallel_tuner.XXXXXX")

cleanup() {
    rm -f "$TUNER_ENV_FILE"
}
trap cleanup EXIT

validate_positive_int() {
    local name=$1
    local value=$2
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$name must be a positive integer, got '$value'."
        exit 1
    fi
}

echo "=========================================="
echo "Starting Benchmark with Configuration:"
echo "=========================================="
echo "MLU Count per Node: $NUM_MLUS"
echo "Batch Size:         $BATCH_SIZE"
echo "Sequence Length:    $SEQ_LEN"
echo "Tensor Parallelism: $TP_DEGREE"
echo "Pipeline Parallelism: $PP_DEGREE"
echo "Data Parallelism:   $DP_DEGREE"
echo "=========================================="


torchrun \
    --nproc-per-node=1 \
    scripts/bench_parallel_tuner.py \
    --num-gpus "$NUM_MLUS" \
    --batch-size "$BATCH_SIZE" \
    --seq-len "$SEQ_LEN" \
    --env-output "$TUNER_ENV_FILE"

source "$TUNER_ENV_FILE"
: "${CRIUS_TUNED_TP_DEGREE:?bench_parallel_tuner.py did not export CRIUS_TUNED_TP_DEGREE}"
: "${CRIUS_TUNED_PP_DEGREE:?bench_parallel_tuner.py did not export CRIUS_TUNED_PP_DEGREE}"
: "${CRIUS_TUNED_DP_DEGREE:?bench_parallel_tuner.py did not export CRIUS_TUNED_DP_DEGREE}"
validate_positive_int "CRIUS_TUNED_TP_DEGREE" "$CRIUS_TUNED_TP_DEGREE"
validate_positive_int "CRIUS_TUNED_PP_DEGREE" "$CRIUS_TUNED_PP_DEGREE"
validate_positive_int "CRIUS_TUNED_DP_DEGREE" "$CRIUS_TUNED_DP_DEGREE"
CRIUS_TUNED_WORLD_SIZE=$((CRIUS_TUNED_TP_DEGREE * CRIUS_TUNED_PP_DEGREE * CRIUS_TUNED_DP_DEGREE))

echo "=========================================="
echo "Tuned Parallelism:"
echo "=========================================="
echo "Tensor Parallelism:   $CRIUS_TUNED_TP_DEGREE"
echo "Pipeline Parallelism: $CRIUS_TUNED_PP_DEGREE"
echo "Data Parallelism:     $CRIUS_TUNED_DP_DEGREE"
echo "World Size:           $CRIUS_TUNED_WORLD_SIZE"
echo "=========================================="

torchrun \
    --nproc-per-node="$CRIUS_TUNED_WORLD_SIZE" \
    scripts/bench_megatron.py \
    --tp-degree "$CRIUS_TUNED_TP_DEGREE" \
    --pp-degree "$CRIUS_TUNED_PP_DEGREE" \
    --dp-degree "$CRIUS_TUNED_DP_DEGREE" \
    --batch-size "$BATCH_SIZE" \
    --seq-len "$SEQ_LEN"

torchrun \
    --nproc-per-node="$NUM_MLUS" \
    scripts/bench_megatron.py \
    --tp-degree "$TP_DEGREE" \
    --pp-degree "$PP_DEGREE" \
    --dp-degree "$DP_DEGREE" \
    --batch-size "$BATCH_SIZE" \
    --seq-len "$SEQ_LEN"

echo "Benchmark completed successfully."

