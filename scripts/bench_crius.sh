#!/bin/bash

NUM_GPUS=$1
BATCH_SIZE=$2
SEQ_LEN=$3
TP_DEGREE=$4
PP_DEGREE=$5
DP_DEGREE=$6

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
    --seq-len "$SEQ_LEN"


# torchrun \
#     --nproc-per-node="$NUM_GPUS" \
#     scripts/bench_megatron.py \
#     --tp-degree "$TP_DEGREE" \
#     --pp-degree "$PP_DEGREE" \
#     --dp-degree "$DP_DEGREE" \
#     --batch-size "$BATCH_SIZE" \
#     --seq-len "$SEQ_LEN"


if [ $? -eq 0 ]; then
    echo "Benchmark completed successfully."
else
    echo "Error: Benchmark failed with exit code $?"
    exit 1
fi
