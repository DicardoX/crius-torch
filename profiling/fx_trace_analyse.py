import argparse
import torch
from torch.fx import symbolic_trace

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Some configuration parameters"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="The index of rounds that need to be omitted.")

    parser.add_argument(
        "--average",
        type=int,
        default=5,
        help="The number of rounds that need to be iterated after warmup.")

    args = parser.parse_args()
    return args

# Simple module for demonstration
class MyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(768, 1536)
        self.linear2 = torch.nn.Linear(1536, 1536)
        self.linear3 = torch.nn.Linear(1536, 768)

    def forward(self, x):
        out = self.linear1(x)
        out = self.linear2(out)
        out = self.linear3(out).clamp(min=0.0, max=1.0)
        return out

def build_model():
    module = MyModule()
    return module

def evaluate_model(model, input_tensor,warmup, average):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for i in range(warmup + average + 1):
        if i == warmup-1:
            start_event.record()

        _ = model(input_tensor)
        torch.cuda.synchronize()

        if i == warmup + average - 1:
            end_event.record()

    forward_time = start_event.elapsed_time(end_event) / average
    print(f"在去掉{warmup}次预热前向计算后，{average}次前向计算的平均时间为: {forward_time} 豪秒")
    return forward_time

def analyze_model(model):

    symbolic_traced: torch.fx.GraphModule = symbolic_trace(model)
    print(symbolic_traced.graph)

    # 遍历 GraphModule 中的每个节点
    for node in symbolic_traced.graph.nodes:
        # 打印当前节点的名称和类型
        print(f"Node: {node.name} ({node.op})")
        # 打印该节点依赖的其他节点
        if node.args:
            print("  Depends on:")
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    print(f"    {arg.name}")
        # 打印依赖该节点的其他节点
        if node.users:
            print("  Used by:")
            for user in node.users:
                print(f"    {user.name}")
        print()

    node_info = []
    for node in symbolic_traced.graph.nodes:
        node_info.append({
            'name': node.name,
            'op': node.op,
            'target': node.target,
            'args': node.args[1:],
            'kwargs': node.kwargs

        })

    return node_info

# 为每个节点构造对应的算子
def create_operator_for_node(node, model, input_shape, device):

    input_tensor = torch.rand(input_shape)
    input_tensor = input_tensor.to(device)

    if node['op'] == 'call_module':
        module = getattr(model, node['target'])
        output = module(input_tensor)
        return module, output.shape

    elif node['op'] == 'call_method':

        class CallMethodOnlyModule(torch.nn.Module):
            def __init__(self, method_name, args, kwargs):
                super().__init__()
                self.method_name = method_name
                self.args = args
                self.kwargs = kwargs

            def forward(self, x):
                method = getattr(x, self.method_name)
                return method(*self.args, **self.kwargs)

        # 构造新的模块
        method_module = CallMethodOnlyModule(node['target'], node['args'], node['kwargs'])
        output = method_module(input_tensor)
        return method_module, output.shape

    elif node['op'] == 'call_function':
        import types
        class SingleOpModule(torch.nn.Module):
            def __init__(self):
                super(SingleOpModule, self).__init__()
                self.op = node['target']

            def forward(self, *args, **kwargs):
                return self.op(*args, **kwargs)

        output_shape = SingleOpModule(input_tensor).shape
        return SingleOpModule(), output_shape
    else:
        return None, input_shape


# 测量节点的执行时间
def measure_node_execution_time(node_op, input_shape, warmup, average, device):

    input_tensor = torch.rand(input_shape)
    input_tensor = input_tensor.to(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for i in range(warmup + average + 1):
        if i == warmup-1:
            start_event.record()

        _ = node_op(input_tensor)
        torch.cuda.synchronize()

        if i == warmup + average - 1:
            end_event.record()

    node_forward_time = start_event.elapsed_time(end_event) / average

    return node_forward_time


if __name__ == "__main__":
    args = _parse_args()

    # 创建模型
    input_tensor = torch.rand(16, 10000, 768)
    model = build_model()

    # 检查 GPU 可用性
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"发现 {num_gpus} 个可用的 GPU.")
        # 可以选择特定的 GPU 设备，这里选择第一个 GPU（索引从 0 开始）
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
    else:
        device = torch.device("cpu")
        print("没有可用的 GPU，使用 CPU 进行计算.")

    # 测量模型前向传播的时间
    input_tensor = input_tensor.to(device)
    model.to(device)
    module_forward_time = evaluate_model(model, input_tensor, warmup=args.warmup, average=args.average)

    # 导出 IR 并遍历节点
    node_info = analyze_model(model)

    # 为每个节点构造对应的算子并测量执行时间
    input_shape = input_tensor.shape
    op_time = []
    for node in node_info:
        node_op, output_shape = create_operator_for_node(node, model, input_shape, device)
        if node_op:
            node_execution_time = measure_node_execution_time(node_op, input_shape, args.warmup, args.average, device)
            print(f"节点 {node['name']} 执行平均时间: {node_execution_time} 豪秒")
            op_time.append(node_execution_time)
        input_shape = output_shape

    single_time_sum = sum(op_time)
    print(f"模型前向计算一次的时间为{module_forward_time}，拆分后所有单算子执行时间的总和为{single_time_sum}，时间相差{module_forward_time-single_time_sum}")


