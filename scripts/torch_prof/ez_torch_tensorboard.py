import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

def profile_with_graph():
    model = models.resnet18().cuda()
    model.eval()  # 设置为评估模式
    
    # 创建输入数据
    inputs = torch.randn(5, 3, 224, 224).cuda()
    
    # 方法2：使用 torch.jit.script 获得更好的图结构
    scripted_model = torch.jit.script(model)

    writer = SummaryWriter('logs/pytorch_fit')
    
    # 预热
    with torch.no_grad():
        for _ in range(3):
            _ = scripted_model(inputs)
    
    # 使用 Profiler 记录计算图
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=1
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            dir_name="logs/graph_profile"
        ),
        record_shapes=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
        # with_graph=True,  # 启用计算图
        profile_memory=True
    ) as prof:
        with torch.no_grad():
            for i in range(5):
                with record_function(f"iteration_{i}"):
                    # 使用更详细的作用域
                    with record_function("forward_pass"):
                        output = scripted_model(inputs)
                    
                    # 添加一些后处理来丰富计算图
                    with record_function("output_processing"):
                        probabilities = torch.softmax(output, dim=1)
                        predicted_class = torch.argmax(probabilities, dim=1)
                
                prof.step()
    
    writer.add_graph(model, inputs)
    writer.close()
    
    return prof

# 运行分析
prof = profile_with_graph()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))