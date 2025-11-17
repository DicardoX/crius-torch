import torch

def test_torch_model_offline_profiling():
    """ Offline profile all operators."""

    from crius.offline.offline_profiler import CriusOfflineDataManager, OfflineProfileConfigs
    profile_configs = OfflineProfileConfigs(
        min_batch_size=2048,
        max_batch_size=2048,
        batch_size_step="power-of-2",
        min_seq_len=4,
        max_seq_len=4,
        seq_len_step="power-of-2",
        # gemm_left_op_hidden_dims =[128, 256, 512, 1024, 2048, 4096, 6144, 8192],
        # gemm_right_op_hidden_dims=[128, 256, 512, 1024, 2048, 4096, 6144, 8192],
        gemm_left_op_hidden_dims =[4096],
        gemm_right_op_hidden_dims=[6144],
        dtype=torch.float16,
        device="cuda",
        num_warmup_steps=5,
        num_profile_steps=10,
        force_overwrite=True,
    )
    offline_profiler = CriusOfflineDataManager(
        profile_configs=profile_configs,
        to_store_path='./script',
        data_filename='ez_prof_data.pkl',
    )
    
    offline_profiler.profile_all_gemm()
    # offline_profiler.profile_all_comm()

if __name__ == '__main__':
    test_torch_model_offline_profiling()