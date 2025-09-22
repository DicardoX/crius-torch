import torch
import torch.fx
from torch.fx import Tracer, GraphModule
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import contextlib

class LayerAwareTracer(Tracer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node_layer_mapping = {}  # Store node to layer class name mapping
        self.current_module_stack = []  # Track current module call stack
        self.module_class_map = {}  # Map module path to class name
    
    def trace(self, root: torch.nn.Module, concrete_args: Optional[Dict[str, Any]] = None) -> GraphModule:
        # Reset state and build module mapping
        self.node_layer_mapping = {}
        self.current_module_stack = []
        self.module_class_map = {}
        
        # Build mapping from module path to class name
        for name, module in root.named_modules():
            self.module_class_map[name] = module.__class__.__name__
        
        # Call parent's trace method
        graph_module = super().trace(root, concrete_args)
        
        # Add layer information to graph_module
        graph_module.node_layer_mapping = self.node_layer_mapping
        return graph_module
    
    @contextlib.contextmanager
    def _module_context(self, module: torch.nn.Module):
        """Context manager for tracking current module"""
        module_path = self.path_of_module(module)
        module_class = module.__class__.__name__
        
        # Push to module stack
        self.current_module_stack.append((module_path, module_class))
        try:
            yield
        finally:
            # Pop from module stack
            if self.current_module_stack:
                self.current_module_stack.pop()
    
    def call_module(self, m: torch.nn.Module, forward: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        with self._module_context(m):
            result = super().call_module(m, forward, args, kwargs)
            
            # Get current created node (usually the last one)
            if hasattr(self, '_graph') and self._graph.nodes:
                current_node = list(self._graph.nodes)[-1]
                # Record node to layer class mapping
                current_path, current_class = self.current_module_stack[-1]
                self._record_node_layer_info(current_node, current_path, current_class)
            
            return result
    
    def create_node(self, *args, **kwargs):
        node = super().create_node(*args, **kwargs)
        
        # If currently in module context, record layer information
        if self.current_module_stack:
            current_path, current_class = self.current_module_stack[-1]
            self._record_node_layer_info(node, current_path, current_class)
        
        return node
    
    def _record_node_layer_info(self, node, module_path, module_class):
        """Record node's layer information"""
        self.node_layer_mapping[node.name] = {
            'layer_class': module_class,
            'module_path': module_path,
            'full_stack': self.current_module_stack.copy(),
            'node_type': node.op
        }
    
    # Override other methods that may create nodes to ensure all nodes are recorded
    def call_function(self, func, args, kwargs):
        result = super().call_function(func, args, kwargs)
        self._record_current_node_if_needed()
        return result
    
    def call_method(self, method_name, args, kwargs):
        result = super().call_method(method_name, args, kwargs)
        self._record_current_node_if_needed()
        return result
    
    def _record_current_node_if_needed(self):
        """Record current node (if exists)"""
        if hasattr(self, '_graph') and self._graph.nodes and self.current_module_stack:
            current_node = list(self._graph.nodes)[-1]
            current_path, current_class = self.current_module_stack[-1]
            self._record_node_layer_info(current_node, current_path, current_class)

# Usage example
def trace_model_with_layer_info(model: torch.nn.Module, example_inputs: Any) -> GraphModule:
    tracer = LayerAwareTracer()
    graph_module = tracer.trace(model)
    
    # Print layer information for each node
    print("Node Layer Information:")
    print("=" * 80)
    for node in graph_module.nodes:
        if node.name in graph_module.node_layer_mapping:
            layer_info = graph_module.node_layer_mapping[node.name]
            print(f"Node: {node.name}")
            print(f"  Op: {node.op}")
            print(f"  Target: {node.target}")
            print(f"  Layer Class: {layer_info['layer_class']}")
            print(f"  Module Path: {layer_info['module_path']}")
            print(f"  Node Type: {layer_info['node_type']}")
            print(f"  In Module Context: {len(layer_info['full_stack']) > 0}")
            print()
        else:
            print(f"Node: {node.name} - No layer information available")
            print()
    
    return graph_module

# Test case
class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        # Multiple operations here belong to linear layer:
        # 1. call_module: linear
        # 2. call_function: matmul (inside linear)
        # 3. call_function: add (bias addition inside linear)
        x = self.linear(x)
        x = self.relu(x)
        return x

if __name__ == "__main__":
    # Create test model
    model = TestModel()
    example_input = torch.randn(2, 10)
    
    # Perform tracing
    traced_model = trace_model_with_layer_info(model, example_input)
    
    print(traced_model)

    # Verify results
    print("Verify all nodes belonging to Linear layer:")
    for node in traced_model.nodes:
        if node.name in traced_model.node_layer_mapping:
            layer_info = traced_model.node_layer_mapping[node.name]
            #if layer_info['layer_class'] == 'Linear':
            print(f" {node.name} ({node.op}) belongs to {layer_info['layer_class']} layer")
    
    # Can also directly access mapping information
    print(f"\nTotal of {len(traced_model.node_layer_mapping)} nodes with layer information tracked")
