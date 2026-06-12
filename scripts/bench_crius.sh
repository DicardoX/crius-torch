#!/bin/bash

NUM_GPUS=$1
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
echo "=========================================="
echo "Starting Benchmark with Configuration:"
echo "=========================================="
echo "GPU Count per Node: $NUM_GPUS"
echo "Batch Size:         $BATCH_SIZE"
echo "Sequence Length:    $SEQ_LEN"
echo "Tensor Parallelism: $TP_DEGREE"
echo "Pipeline Parallelism: $PP_DEGREE"
echo "Data Parallelism:   $DP_DEGREE"
echo "=========================================="


torchrun \
    --nproc-per-node=1 \
    scripts/bench_parallel_tuner.py \
    --num_gpus "$NUM_GPUS" \
    --batch-size "$BATCH_SIZE" \
    --seq-len "$SEQ_LEN" \
    --env-output "$TUNER_ENV_FILE"
source "$TUNER_ENV_FILE"
: "${CRIUS_TUNED_TP_DEGREE:?bench_parallel_tuner.py did not export CRIUS_TUNED_TP_DEGREE}"
: "${CRIUS_TUNED_PP_DEGREE:?bench_parallel_tuner.py did not export CRIUS_TUNED_PP_DEGREE}"
: "${CRIUS_TUNED_DP_DEGREE:?bench_parallel_tuner.py did not export CRIUS_TUNED_DP_DEGREE}"
echo "=========================================="
echo "Tuned Parallelism:"
echo "=========================================="
echo "Tensor Parallelism:   $CRIUS_TUNED_TP_DEGREE"
echo "Pipeline Parallelism: $CRIUS_TUNED_PP_DEGREE"
echo "Data Parallelism:     $CRIUS_TUNED_DP_DEGREE"
echo "=========================================="
CRIUS_TUNED_WORLD_SIZE=$((CRIUS_TUNED_TP_DEGREE * CRIUS_TUNED_PP_DEGREE * CRIUS_TUNED_DP_DEGREE))
echo "CRIUS_TUNED_WORLD_SIZE:     $CRIUS_TUNED_WORLD_SIZE"
torchrun \
    --nproc-per-node="$CRIUS_TUNED_WORLD_SIZE" \
    scripts/bench_megatron.py \
    --tp-degree "$CRIUS_TUNED_TP_DEGREE" \
    --pp-degree "$CRIUS_TUNED_PP_DEGREE" \
    --dp-degree "$CRIUS_TUNED_DP_DEGREE" \
    --batch-size "$BATCH_SIZE" \
    --seq-len "$SEQ_LEN"

torchrun \
    --nproc-per-node="$NUM_GPUS" \
    scripts/bench_megatron.py \
    --tp-degree "$TP_DEGREE" \
    --pp-degree "$PP_DEGREE" \
    --dp-degree "$DP_DEGREE" \
    --batch-size "$BATCH_SIZE" \
    --seq-len "$SEQ_LEN"
#echo "Average end-to-end iteration time (ms): 999.94175"

if [ $? -eq 0 ]; then
    echo "Benchmark completed successfully."
else
    echo "Error: Benchmark failed with exit code $?"
    exit 1
fi
