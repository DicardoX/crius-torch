import logging
import re
import torch
from typing import List, Tuple, Set
from crius.compile.graph import IRGraph, IRNode
# from crius.compile.compiler import CompileConfigs
from crius.model.model_config import model_config
from crius.model_parallel.utils import ParallelismPlan


'''
QWen3 model example, ep only partition the expert module in mlp , the gate in different ep group is copy
>>> model Qwen3MoeForCausalLM(
  (model): Qwen3MoeModel(
    (embed_tokens): Embedding(151936, 2048)
    (layers): ModuleList(
      (0-47): 48 x Qwen3MoeDecoderLayer(
        (self_attn): Qwen3MoeAttention(
          (q_proj): Linear(in_features=2048, out_features=4096, bias=False)
          (k_proj): Linear(in_features=2048, out_features=512, bias=False)
          (v_proj): Linear(in_features=2048, out_features=512, bias=False)
          (o_proj): Linear(in_features=4096, out_features=2048, bias=False)
          (q_norm): Qwen3MoeRMSNorm((128,), eps=1e-06)
          (k_norm): Qwen3MoeRMSNorm((128,), eps=1e-06)
        )
        (mlp): Qwen3MoeSparseMoeBlock(
          (gate): Linear(in_features=2048, out_features=128, bias=False)
          (experts): ModuleList(
            (0-127): 128 x Qwen3MoeMLP(
              (gate_proj): Linear(in_features=2048, out_features=768, bias=False)
              (up_proj): Linear(in_features=2048, out_features=768, bias=False)
              (down_proj): Linear(in_features=768, out_features=2048, bias=False)
              (act_fn): SiLU()
            )
          )
        )
        (input_layernorm): Qwen3MoeRMSNorm((2048,), eps=1e-06)
        (post_attention_layernorm): Qwen3MoeRMSNorm((2048,), eps=1e-06)
      )
    )
    (norm): Qwen3MoeRMSNorm((2048,), eps=1e-06)
    (rotary_emb): Qwen3MoeRotaryEmbedding()
  )
  (lm_head): Linear(in_features=2048, out_features=151936, bias=False)
)

'''


'''
通过对计算图进行分析，一个专家的算子应该包括如下：
因为现在实现中认为getitem计算开销小，忽略了计算开销，因此实际解析计算图可以任务两个where算子之间的部分是一个专家的所有算子
------------------
	
    %getitem_59 : [num_users=1] = call_function[target=operator.getitem](args = (%expert_mask, 0), kwargs = {})
    %where : [num_users=3] = call_function[target=torch.where](args = (%getitem_59,), kwargs = {})
    %getitem_32754 : [num_users=1] = call_function[target=operator.getitem](args = (%where, 0), kwargs = {})
    %sym_size_int_1 : [num_users=2] = call_function[target=torch.ops.aten.sym_size.int](args = (%getitem_32754, 0), kwargs = {})
    %sym_constrain_range_for_size_default : [num_users=0] = call_function[target=torch.ops.aten.sym_constrain_range_for_size.default](args = (%sym_size_int_1,), kwargs = {})
    %ge : [num_users=1] = call_function[target=operator.ge](args = (%sym_size_int_1, 0), kwargs = {})
    %_assert_scalar_default : [num_users=0] = call_function[target=torch.ops.aten._assert_scalar.default](args = (%ge, Runtime assertion failed for expression u0 >= 0 on node 'ge'), kwargs = {})
    %idx : [num_users=1] = call_function[target=operator.getitem](args = (%where, 0), kwargs = {})
    %top_x : [num_users=3] = call_function[target=operator.getitem](args = (%where, 1), kwargs = {})
    %getitem_62 : [num_users=1] = call_function[target=operator.getitem](args = (%hidden_states_13, (None, %top_x)), kwargs = {})
    %current_state : [num_users=2] = call_method[target=reshape](args = (%getitem_62, -1, 2048), kwargs = {})
    %l__self___model_layers_0_mlp_experts_0_gate_proj : [num_users=1] = call_module[target=L__self___model_layers_0_mlp_experts_0_gate_proj](args = (%current_state,), kwargs = {})
    %l__self___model_layers_0_mlp_experts_0_act_fn : [num_users=1] = call_module[target=L__self___model_layers_0_mlp_experts_0_act_fn](args = (%l__self___model_layers_0_mlp_experts_0_gate_proj,), kwargs = {})
    %l__self___model_layers_0_mlp_experts_0_up_proj : [num_users=1] = call_module[target=L__self___model_layers_0_mlp_experts_0_up_proj](args = (%current_state,), kwargs = {})
    %mul_15 : [num_users=1] = call_function[target=operator.mul](args = (%l__self___model_layers_0_mlp_experts_0_act_fn, %l__self___model_layers_0_mlp_experts_0_up_proj), kwargs = {})
    %down_proj : [num_users=1] = call_module[target=L__self___model_layers_0_mlp_experts_0_down_proj](args = (%mul_15,), kwargs = {})
    %getitem_63 : [num_users=1] = call_function[target=operator.getitem](args = (%routing_weights_3, (%top_x, %idx, None)), kwargs = {})
    %current_hidden_states : [num_users=1] = call_function[target=operator.mul](args = (%down_proj, %getitem_63), kwargs = {})
    %to_12 : [num_users=1] = call_method[target=to](args = (%current_hidden_states, torch.float16), kwargs = {})
    %index_add_ : [num_users=0] = call_method[target=index_add_](args = (%final_hidden_states, 0, %top_x, %to_12), kwargs = {})

	
---------------------------------	

'''

import re
from typing import List, Tuple, Dict, Optional
import logging

def analyze_expert_structure(graph: IRGraph) -> Dict[str, object]:
    """
    分析 MoE 模型中每个专家的算子结构，以 arange_<id> 为起始标记。
    最后一个专家的范围由前几个专家的算子数量决定。
    """
    # 匹配 arange_<数字> 的正则表达式
    arange_pattern = re.compile(r"^arange_(\d+)$")

    # 收集所有 arange 节点及其在图中的顺序
    arange_nodes: List[Tuple[int, object]] = []  # (expert_id, node)

    for node in graph.nodes:
        match = arange_pattern.match(str(node.name))
        if match:
            expert_id = int(match.group(1))
            arange_nodes.append((expert_id, node))

    # 按照在图中出现的顺序排序
    ordered_arange_nodes = sorted(arange_nodes, key=lambda x: list(graph.nodes).index(x[1]))
    arange_node_list = [node for _, node in ordered_arange_nodes]

    num_experts = len(arange_node_list)
    if num_experts == 0:
        logging.warning("No 'arange_<id>' nodes found in the graph.")
        return {
            "is_valid": False,
            "num_experts": 0,
            "op_counts": [],
            "consistent": True,
            "expert_ranges": []
        }

    if num_experts == 1:
        logging.warning("Only one arange_<id> node found. Cannot determine op count from peers.")
        # 无法统计，返回空结构
        return {
            "is_valid": False,
            "num_experts": 1,
            "op_counts": [None],
            "consistent": False,
            "expert_ranges": []
        }
    
    # Step 1: 获取每个arange节点的前一个节点作为实际起始
    expert_starts = []
    for arange_node in arange_node_list:
        prev_node = arange_node.prev
        if prev_node is None:
            logging.error(f"arange node '{arange_node.name}' has no predecessor. Cannot determine start.")
            return {
                "is_valid": False,
                "num_experts": num_experts,
                "op_counts": [],
                "consistent": False,
                "expert_ranges": []
            }
        expert_starts.append(prev_node)

    # Step 1: 统计前 N-1 个专家的算子数量（每个从 arange_i 到 arange_{i+1} 之前）
    op_counts: List[int] = []
    for i in range(num_experts - 1):
        start_node = expert_starts[i]
        end_node = expert_starts[i + 1]  # 不包含

        count = 0
        current = start_node
        while current is not None and current != end_node:
            count += 1
            current = current.next

        op_counts.append(count)

    # Step 2: 检查前 N-1 个专家的算子数量是否一致
    base_op_count = op_counts[0]
    if not all(c == base_op_count for c in op_counts):
        logging.warning(f"Expert op counts are inconsistent: {op_counts}. Cannot partition reliably.")
        return {
            "is_valid": False,
            "num_experts": num_experts,
            "op_counts": op_counts,
            "consistent": False,
            "expert_ranges": []
        }

    # Step 3: 为最后一个专家构建范围：从 arange_last 开始，往后数 base_op_count 个节点
    last_start = expert_starts[-1]
    last_nodes = []
    current = last_start
    for _ in range(base_op_count):
        if current is None:
            break
        last_nodes.append(current)
        current = current.next

    if len(last_nodes) != base_op_count:
        logging.warning(
            f"Last expert does not have enough nodes: expected {base_op_count}, got {len(last_nodes)}."
        )
        return {
            "is_valid": False,
            "num_experts": num_experts,
            "op_counts": op_counts + [len(last_nodes)],
            "consistent": False,
            "expert_ranges": []
        }

    # Step 4: 构建所有专家的范围
    expert_ranges = []
    # 前 N-1 个专家
    for i in range(num_experts - 1):
        start = expert_starts[i]
        end = expert_starts[i + 1]  # exclusive
        nodes = []
        current = start
        while current != end:
            nodes.append(current)
            current = current.next
        expert_ranges.append({
            "expert_id": i,
            "start_node": start,
            "end_node_exclusive": end,
            "nodes": nodes,
            "count": len(nodes)
        })

        # 打印当前专家的算子列表
        print(f"\n 🧩 Expert {i} ({len(nodes)} ops):")
        for idx, node in enumerate(nodes):
            print(f"  {idx:2d}. {node.name} | {str(node.target)}")

    # 最后一个专家
    expert_ranges.append({
        "expert_id": num_experts - 1,
        "start_node": last_start,
        "end_node_exclusive": current,  # 当前停止的位置
        "nodes": last_nodes,
        "count": len(last_nodes)
    })

    # ✅ 打印最后一个专家的算子列表
    print(f"\n🧩 Expert {num_experts - 1} ({len(last_nodes)} ops) [Last Expert]:")
    for idx, node in enumerate(last_nodes):
        print(f"  {idx:2d}. {node.name} | {str(node.target)}")

    logging.info(f"All {num_experts} experts have {base_op_count} ops. Structure is consistent.")

    return {
        "is_valid": True,
        "num_experts": num_experts,
        "op_counts": op_counts + [base_op_count],
        "consistent": True,
        "base_op_count": base_op_count,
        "expert_ranges": expert_ranges
    }

def partition_graph_by_ep_parallel(
        graph: IRGraph, 
        total_experts: int,
        ep_degree: int,
        current_ep_rank: int
    ) -> IRGraph:

    # Step 1:  参数配置校验和Moe模型算子分析
    if model_config.if_moe is False:
        logging.warning(f"This is not a moe model...")
        return graph
    
    if total_experts % ep_degree != 0:
        raise ValueError(f"Total experts ({total_experts}) must be devisible by expert parallel degree ({ep_degree})")
    
    analysis_result = analyze_expert_structure(graph)

    if not analysis_result.get("is_valid", False):
        logging.error("Expert structure analysis failed, Cannot partition graph.")
        return graph
    
    expert_ranges = analysis_result.get("expert_ranges", [])
    if not isinstance(expert_ranges, list):
        raise TypeError(f"Expected 'expert_ranges' to be a list, but got {type(expert_ranges)}" )
    
    if len(expert_ranges) == 0:
        logging.error("No expert ranges found, nothing to partition.")
        return graph

    num_local_experts = total_experts // ep_degree
    start_idx = current_ep_rank * num_local_experts
    end_idx = start_idx + num_local_experts

    logging.info(
        f"Expert Parallel Partitioning: "
        f"Total expert={total_experts}, EP degree={ep_degree}, "
        f"Local experts = {num_local_experts}, "
        f"Rank {current_ep_rank} owns experts [{start_idx}, {end_idx}]"
    )

    # Step 2: 根据当前的ep rank确定需要保留的专家节点
    nodes_to_keep: Set[object] = set()
    for exp_id in range(start_idx, end_idx):
        if exp_id < len(expert_ranges):
            nodes_to_keep.update(expert_ranges[exp_id]["nodes"])
        else:
            logging.warning(f"Expert ID {exp_id} exceeds available expert_ranges length {len(expert_ranges)}.")

    # Step 3: 确定需要删除的节点（不属于当前 rank 的专家节点）
    nodes_to_remove: Set[object] = set() 
    for exp_id in range(total_experts):
        if exp_id < len(expert_ranges) and exp_id not in range(start_idx, end_idx):
            exp_range = expert_ranges[exp_id]
            if isinstance(exp_range, dict) and "nodes" in exp_range:
                nodes_list = exp_range["nodes"]
                if isinstance(nodes_list, list):
                    nodes_to_remove.update(nodes_list)
                else:
                    logging.warning(f"Invalid 'nodes' type for expert {exp_id}: {type(nodes_list)}")
            else:
                logging.warning(f"Invalid exp_range for expert {exp_id}")

    # Step 4: 获取图中所有节点的顺序
    all_nodes_in_order = list(graph.nodes) 

    # Step 5: 按照原图顺序对要删除的节点进行排序，然后反向
    sorted_nodes_to_remove = [n for n in all_nodes_in_order if n in nodes_to_remove]
    reversed_nodes_to_remove = sorted_nodes_to_remove[::-1] # 反向

    # Step 7: 反向删除节点
    for node in reversed_nodes_to_remove:
        try:
            # 检查是否有下游节点使用它
            # 注意：即使我们反向删除，也要检查，因为可能存在复杂的依赖或 IRGraph 内部的特殊处理
            # torch.fx 中，node.users 是一个包含使用该节点输出的下游节点的字典
            # 如果 node.users 为空，说明没有节点依赖它，可以安全删除
            # 如果不为空，说明仍有依赖，这在反向删除中理论上不应该发生，但为了安全起见，检查一下
            if hasattr(node, 'users') and node.users:
                # 记录警告
                logging.warning(f"Node {node.name} still has {len(node.users)} users during reverse deletion: {[u.name for u in node.users.keys()]}. Attempting removal anyway, this might fail.")
            graph.erase_node(node)
        except Exception as e:
            logging.error(f"Failed to remove node {node.name} during reverse deletion: {e}")

    # Step 8: 清理图结构， todo ？？ 是否需要
    # graph.liveness_analysis()  # 如有此方法
    # graph.cleanup()
    logging.info(f"Expert parallel partitioning completed for rank {current_ep_rank}.")
    return graph     


def insert_ep_comm_op(
        graph: IRGraph
) -> IRGraph:
    
    analysis_result = analyze_expert_structure(graph)

    if not analysis_result.get("is_valid", False):
        logging.error("Expert structure analysis failed, Cannot partition graph.")
        return graph
    
    expert_ranges = analysis_result.get("expert_ranges", [])
    if not isinstance(expert_ranges, list):
        raise TypeError(f"Expected 'expert_ranges' to be a list, but got {type(expert_ranges)}" )
    
    if len(expert_ranges) == 0:
        logging.error("No expert ranges found, nothing to partition.")
        return graph
    
    all_expert_nodes: dict[object, None] = {}  
    for exp_range in expert_ranges:
        if not isinstance(exp_range, dict) or "nodes" not in exp_range:
            continue
        nodes_list = exp_range["nodes"]
        if not isinstance(nodes_list, list):
            continue
        for n in nodes_list:
            all_expert_nodes.setdefault(n, None)   

    ordered_expert_nodes = list(all_expert_nodes)
    
    # --- Automatically find insertion points ---
    dispatch_insert_after_node_name, combine_insert_after_node_name = find_insertion_points(
        graph, ordered_expert_nodes
    )
    return graph

    # --- Insert Dispatch (if location found), TODO ---
    dispatch_node_name = None
    if dispatch_insert_after_node_name:
        original_hidden_states_name = None # TODO: Identify from graph context if needed

        try:
            dispatch_node_name = insert_dispatch_or_combine_comm_op(
                graph=graph,
                insert_after_node_name=dispatch_insert_after_node_name,
                op_type='dispatch',
                original_hidden_states_node=original_hidden_states_name
            )
            logging.info(f"Successfully inserted dispatch op: {dispatch_node_name}")
        except Exception as e:
            logging.error(f"Failed to insert dispatch op after '{dispatch_insert_after_node_name}': {e}")
            exit(0)
            # Depending on requirements, you might want to fail here or continue without dispatch
    else:
        logging.warning("Could not determine dispatch insertion point. Skipping dispatch op insertion.")

    # return graph

    # --- Insert Combine (if location found),TODO ---
    combine_node_name = None
    if combine_insert_after_node_name:
        try:
            combine_node_name = insert_dispatch_or_combine_comm_op(
                graph=graph,
                insert_after_node_name=combine_insert_after_node_name,
                op_type='combine'
            )
            logging.info(f"Successfully inserted combine op: {combine_node_name}")            

        except Exception as e:
            logging.error(f"Failed to insert combine op after '{combine_insert_after_node_name}' or update residual add: {e}")
            # Depending on requirements, you might want to fail here or continue without combine
    else:
        logging.warning("Could not determine combine insertion point. Skipping combine op insertion.")

def insert_dispatch_or_combine_comm_op(
        graph: IRGraph,
        insert_after_node_name: str,
        op_type: str, # 'dispatch' or 'combine'
        original_hidden_states_node: Optional[str] = None # Name of original hidden_states node for dispatch
    ):
    """
    Inserts a dispatch or combine communication operator into the graph.

    Args:
        graph: The IRGraph object.
        insert_after_node_name: The name of the node after which to insert the op.
        op_type: The type of communication op, either 'dispatch' or 'combine'.
        routing_info_nodes: Optional list of node names containing routing information (e.g., selected_experts).
                            Needed for dispatch.
        original_hidden_states_node: Name of the original hidden_states node. Needed for dispatch.

    Returns:
        The name of the newly created communication op node.
    """
   return 


def find_insertion_points(
    graph: 'IRGraph',
    all_expert_nodes: List[object]
) -> Tuple[Optional[str], Optional[str]]:
    """
    Automatically finds the insertion points for dispatch and combine.

    Args:
        graph: The IRGraph object.
        all_expert_nodes: Set of all nodes identified as belonging to experts.

    Returns:
        A tuple (dispatch_insert_after_node_name, combine_insert_after_node_name).
        Returns None for a point if it cannot be determined.
    """
    all_nodes_in_order = list(graph.nodes)

    # --- Find Dispatch Insertion Point ---
    dispatch_insert_after_node = None
    # Iterate through the graph nodes in order
    for node in all_nodes_in_order:
        # Check if this node is the *first* node that belongs to any expert range
        if node in all_expert_nodes:
            # The dispatch op should be inserted *before* this first expert node.
            # So, we look for the node *just before* this first expert node.
            node_index = all_nodes_in_order.index(node)
            if node_index > 0:
                dispatch_insert_after_node = all_nodes_in_order[node_index - 1].name
                logging.info(f"Found dispatch insertion point after node: {dispatch_insert_after_node}")
            else:
                logging.warning("First node in graph is an expert node. Cannot insert dispatch before it.")
            break # Found the first expert node, stop searching

    # If no expert nodes were found in the order, dispatch insertion point is unclear
    if dispatch_insert_after_node is None:
        logging.warning("No expert nodes found in graph order, or first node is an expert. Dispatch insertion point not found.")

    # --- Find Combine Insertion Point ---
    combine_insert_after_node = None
    last_expert_node = None
    for node in reversed(all_nodes_in_order):
        for expert_node in all_expert_nodes:
            if str(node.name) == str(expert_node.name):
                last_expert_node = node
                break
        else:
            continue
        break
    
    if last_expert_node:
        last_expert_index = all_nodes_in_order.index(last_expert_node)
        combine_target_index = last_expert_index + 2

        if combine_target_index < len(all_nodes_in_order):
            combine_insert_after_node = all_nodes_in_order[combine_target_index].name
            logging.info(f"Found combine insertion point after node: "
                         f"{combine_insert_after_node} (target node index: {combine_target_index})")
        elif last_expert_index + 1 < len(all_nodes_in_order):
             # If i+2 is out of bounds, but i+1 exists, combine after i+1
             combine_insert_after_node = all_nodes_in_order[last_expert_index + 1].name
             logging.info(f"Combine insertion point (fallback) after node: "
                          f"{combine_insert_after_node} (target node index: {last_expert_index + 1})")
        else:
             # If i+1 is also out of bounds, combine after the last expert node itself
             combine_insert_after_node = last_expert_node.name
             logging.info(f"Combine insertion point (fallback) after last expert node: "
                          f"{combine_insert_after_node} (target node index: {last_expert_index})")
    else:
        logging.warning("No expert nodes found in graph order. Cannot determine combine insertion point.")

    return dispatch_insert_after_node, combine_insert_after_node

def insert_dispatch_or_combine_comm_op_2(node: IRNode, ep_degree:int):
    
    '''
    dispatch: 将输入的 hidden_states (形状 [Batch_Size, Seq_Length, Model_Dim] 或合并为 [Total_Tokens, Model_Dim]) 
    根据路由结果，分发给全局的各个专家。每个 Rank 只接收并处理分配给该 Rank 本地专家的那部分 tokens。

    因此，我们从hidden states算子获取dispatch 的通信量
    每个rank发送的总数据量 (total_tokens / ep_degree ) ( model_dim *dtype_size)
    每个rank接收的总数据量 = 发送
    '''

    '''
    combine: 将各个 Rank 上本地专家计算出的输出，根据原始路由结果，重新聚合回 [Total_Tokens, Model_Dim] 的形状，恢复到原始 token 的顺序。   

    每个rank发送的数据量:  (Total_Tokens / EP_Degree) * Model_Dim * DType_Size
    每个rank接收的总数据量 = 发送

    combine的通信量==dispatch的通信量
    '''

    '''
    dispatch 算子的插入位置：应该在路由逻辑完成之后，专家计算开始之前，因此插入在 %expert_mask 节点之后
    combine 算子的插入位置：应该在所有专家计算完成之后，但在残差连接(hidden_states_719)之前进行。因此可以插入在final_hidden_states_95 算子之后
    '''

    ''' hidden_states_13 op example
    DEBUG:crius.global_envs:=====> Node target: <built-in function mul>
    DEBUG:crius.global_envs:Node name: hidden_states_12 | OP type: call_function | dtype: torch.bfloat16 | 
    Next: hidden_states_13 | Input shapes: [torch.Size([2048]), torch.Size([2, 40960, 2048])] | 
    Output shapes: [torch.Size([2, 40960, 2048])] | Latency: 1.1808768272399903
    '''
    output_shapes = node.output_shapes
    total_token = output_shapes[0][1]
    model_dim = output_shapes[0][2]
    dtype = node.dtype

    # 数据类型对应的字节数
    dtype_to_bytes = {
        'torch.float32': 4,
        'torch.float16': 2,
        'torch.bfloat16': 2,
        'torch.int8': 1
    }
    dtype_size = dtype_to_bytes[dtype]

    comm_size = (total_token // ep_degree) * model_dim * dtype_size
    print(f"The expert parallel dispatch or combine comm size is {comm_size}")

    return