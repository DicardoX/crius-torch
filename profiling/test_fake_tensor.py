import argparse, contextlib, gc, time, psutil, torch, transformers
from torch._subclasses.fake_tensor import FakeTensorMode
from tabulate import tabulate   # pip install tabulate

proc   = psutil.Process()

# ──────────────────────────── helpers ──────────────────────────────
def now_cpu_mb() -> float:
    """Current RSS (resident set size) in MB."""
    return proc.memory_info().rss / 1e6

def peak_gpu_mb(dev: str) -> float:
    """Peak allocated GPU memory since last reset (MB)."""
    return torch.cuda.max_memory_allocated(dev) / 1e6

@contextlib.contextmanager
def measure(tag: str, device: str):
    """Context manager that times a code block & tracks CPU / GPU peaks."""
    torch.cuda.reset_peak_memory_stats(device)
    cpu_before = now_cpu_mb()
    t0 = time.perf_counter()
    yield
    torch.cuda.synchronize(device)
    dt   = time.perf_counter() - t0
    cpu_peak = max(now_cpu_mb(), cpu_before) - cpu_before
    gpu_peak = peak_gpu_mb(device)
    results[tag] = dict(time_s=dt, cpu_mb=cpu_peak, gpu_mb=gpu_peak)

# ──────────────────────────── benchmarks ───────────────────────────
def load_normal(model_path: str, device: str, dtype):
    # 确保local_files_only=True参数被传递给所有from_pretrained调用
    config = transformers.AutoConfig.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True)
    model  = transformers.AutoModelForCausalLM.from_config(
        config, torch_dtype=dtype)
    model.to(device)          # materialise
    return model

def load_fake(model_path: str, device: str, dtype):
    fake_mode = FakeTensorMode()
    with fake_mode:
        # 确保local_files_only=True参数被传递给所有from_pretrained调用
        config = transformers.AutoConfig.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True)
        model  = transformers.AutoModelForCausalLM.from_config(
            config, torch_dtype=dtype)
    model.to(device)          # materialise
    return model

# ──────────────────────────── main ────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True, help="local HF model path")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(0)
    model_name = args.model.split("/")[-1]
    device = args.device
    dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    global results; results = {}

    with measure("normal", device):
        m = load_normal(args.model, device, dtype)
    del m; gc.collect(); torch.cuda.empty_cache()

    with measure("fake", device):
        m = load_fake(args.model, device, dtype)
    del m; gc.collect(); torch.cuda.empty_cache()

    # ── print table ──
    print(f"\n=== Load benchmark for {model_name} ===")
    print(tabulate(
        [(k, f"{v['time_s']:.2f}", f"{v['cpu_mb']:.1f}", f"{v['gpu_mb']:.1f}")
         for k, v in results.items()],
        headers=["mode", "time (s)", "CPU ΔRSS (MB)", "GPU peak (MB)"],
        tablefmt="github"))

if __name__ == "__main__":
    # python test_fake_tensor.py --model /app/profiling/model/qwen-7b-chat --device cuda:0
    main()