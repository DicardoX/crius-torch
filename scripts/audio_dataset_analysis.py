import os
import torch
from datasets import load_dataset, DownloadConfig
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2Config, Wav2Vec2ConformerForCTC
from tqdm import tqdm
import pandas as pd

os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'  # 启用hf_transfer加速大文件下载

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载预训练的Conformer处理器和随机初始化模型
# 创建配置（不加载预训练参数）
config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large-960h-ft")

# 修改配置参数
config.feat_proj_dropout = 0.1
config.attention_dropout = 0.1
config.hidden_dropout = 0.1
config.final_dropout = 0.1
config.layerdrop = 0.1
config.pooling_mode = "mean"  # 可选："mean", "max", "cls"

# 加载处理器（仍需预训练的分词器配置）
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large-960h-ft")

# 随机初始化模型（不加载预训练权重）
model = Wav2Vec2ConformerForCTC(config).to(device)

download_config = DownloadConfig(
    num_proc=8,  # 并行下载进程数
    max_retries=3,  # 最大重试次数
    resume_download=True  # 支持断点续传
)

# 以streaming方式加载LibriSpeech数据集
# 可以选择其他子集如'train-clean-100', 'train-clean-360', 'train-other-500'等
dataset = load_dataset("librispeech_asr", "clean", split="validation", trust_remote_code=True)

# 创建保存结果的目录
os.makedirs("results", exist_ok=True)

# 初始化结果列表
results = []

# 处理数据集样本
print("开始处理数据集样本...")
for i, sample in enumerate(dataset):
    print(f"===========> Sample {i}")

    print(sample)

    # 获取音频数据
    audio = sample["audio"]["array"]
    sampling_rate = sample["audio"]["sampling_rate"]

    # 准备输入
    inputs = processor(audio, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

    # 将输入移至设备
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # 进行编码
    with torch.no_grad():
        outputs = model(input_values, attention_mask=attention_mask)

    # 获取logits并计算token数量
    logits = outputs.logits
    predicted_ids = torch.argmax(logits, dim=-1)
    token_count = (predicted_ids != processor.tokenizer.pad_token_id).sum().item()

    # 收集结果
    results.append({
        "sample_id": i,
        "transcript": sample["text"],
        "audio_length": len(audio) / sampling_rate,  # 音频长度（秒）
        "token_count": token_count
    })

    # # 每处理100个样本保存一次结果
    # if (i + 1) % 100 == 0:
    #     df = pd.DataFrame(results)
    #     df.to_csv(f"results/librispeech_token_counts_{i+1}.csv", index=False)

# 保存最终结果
df = pd.DataFrame(results)
df.to_csv("results/librispeech_token_counts_final.csv", index=False)

# 打印统计摘要
print("\n处理完成！统计摘要：")
print(f"总共处理样本数: {len(results)}")
print(f"平均token数量: {df['token_count'].mean():.2f}")
print(f"最小token数量: {df['token_count'].min()}")
print(f"最大token数量: {df['token_count'].max()}")
