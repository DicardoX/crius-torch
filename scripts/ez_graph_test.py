from torch.fx import Graph, Node
import torch

def create_graph_manually():
    
    graph = Graph()
    
    input_node = graph.placeholder('x')
    
    relu_node = graph.call_function(torch.relu, args=(input_node,))
    
    linear_module = torch.nn.Linear(5, 3)
    linear_node = graph.call_module('linear', args=(relu_node,))
    
    graph.output(linear_node)
    
    return graph

manual_graph = create_graph_manually()
print("Manually created graph:")
for node in manual_graph.nodes:
    print(f"  {node.op}: {node.name}")
