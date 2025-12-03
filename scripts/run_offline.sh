#!/bin/bash
# Simplified communication performance profiling script

set -e  # Exit on error

echo "Starting communication performance profiling..."

echo "=== Phase 1: 1-process analysis ==="
# torchrun --nnodes=1 --nproc_per_node=1 /app/scripts/test_torch_model_profile.py

echo "=== Phase 2: 2-process analysis ==="
torchrun --nnodes=1 --nproc_per_node=2 /app/scripts/test_torch_model_profile.py

echo "=== Phase 3: 4-process analysis ==="  
torchrun --nnodes=1 --nproc_per_node=4 /app/scripts/test_torch_model_profile.py

echo "All profiling completed!"