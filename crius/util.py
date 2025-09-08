import pynvml
import re
import os

def get_filename_include_path():
    store_path, filename = get_comp_file()
    return store_path + filename

def get_comp_file():
    to_store_path='./crius/offline/'
    org_data_filename='prof_data.pkl'

    gpu_model = get_gpu_info()
    gpu_keyword = get_gpu_keyword(gpu_model)

    print(gpu_keyword)

    data_file = gpu_keyword + '_' + org_data_filename
    return to_store_path, data_file

def get_gpu_keyword(gpu_model: str):

    series_patterns = [
        r'\b(RTX|GTX|Quadro)\s+[0-9]+[A-Za-z]*\b',  # 匹配 RTX 4090, GTX 3080
        r'\b([A-Za-z]+[0-9]+[A-Za-z]*)\b'           # 匹配 A100, L4, H100, V100
    ]

    # 去除无关前缀（如 "NVIDIA", "GPU:", "Model:" 等）
    cleaned = re.sub(r'^\s*(NVIDIA|GeForce|Tesla|Quadro|GPU|Model|:)\s*', '', gpu_model, flags=re.IGNORECASE)

    for pattern in series_patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            gpu_keyword= match.group(0).strip()
    
    # 如果没有匹配，尝试找任何“字母+数字”或“数字+字母”的组合词
    match = re.search(r'\b\w*[A-Za-z][0-9]+\w*|\w*[0-9][A-Za-z]\w*\b', cleaned)
    if match:
        gpu_keyword= match.group(0).strip()

    gpu_keyword = gpu_keyword.lower().replace(' ', '_')
    return gpu_keyword    

def get_gpu_info():
    try:
        # 初始化 NVML 库
        pynvml.nvmlInit()

        # get cuda_visible_devices
        visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
        if visible_devices is None:
            physical_id = 0 
        else:
            # 解析为物理 GPU ID 列表
            physical_ids = [int(i.strip()) for i in visible_devices.split(",")]
            physical_id = physical_ids[0]
        print(physical_id)
        # 获取第一个 GPU 的句柄
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_id)

        # 获取 GPU 型号名称
        gpu_model = pynvml.nvmlDeviceGetName(handle)

        # 判断是否是字符串
        if isinstance(gpu_model, str):
            gpu_model = gpu_model
        elif isinstance(gpu_model, bytes):
            gpu_model = gpu_model.decode('utf-8')
            print(f"GPU Model: {gpu_model}")
        else:
            print(f"gpu_model type is  {type(gpu_model)} , not str")
        # .decode('utf-8')
        # 获取显存信息（总显存）
        # memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # total_memory_mb = memory_info.total / (1024 ** 2)  # 转换为 MB
        # 关闭 NVML 库
        pynvml.nvmlShutdown()
        return gpu_model
    
    except Exception as e:
        print("Error occurred: {e}")
        return None, None 
