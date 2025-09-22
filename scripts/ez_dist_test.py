import torch.distributed as dist
from crius.train_backend.megatron.utils import initialize_distributed

initialize_distributed(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
)
print("Initialization successful")
