"""Subgraph serialization — converts Python Subgraph objects to JSON for C++ consumption.

Note: Per the design, C++ compile_and_run() returns List[py::array_t<float>] directly
via pybind11 (no JSON/base64 round-trip needed). So deserialize_result is NOT needed.
"""

import json
from typing import Any, Dict, List, Tuple

from .graph import Graph, GraphNode
from .partition import Subgraph


def _node_to_dict(
    node: GraphNode,
    node_id_to_index: Dict[int, int],
    input_node_id_to_arg: Dict[int, int] | None = None,
    num_nodes: int = 0,
) -> Dict[str, Any]:
    """Convert a single GraphNode to a JSON-serializable dict.

    Args:
        node: The graph node to serialize
        node_id_to_index: Mapping from global node_id to local index within the subgraph
        input_node_id_to_arg: Mapping from external input node_id to function argument index
                              (arg_index = num_nodes + position_in_input_mapping)
        num_nodes: Number of nodes in the subgraph (used to offset external input indices)

    Returns:
        Dict with keys: op, inputs, attrs, shape, dtype
    """
    inputs = []
    for inp in node.inputs:
        if inp.id in node_id_to_index:
            # Internal reference: local node index [0, num_nodes)
            inputs.append(node_id_to_index[inp.id])
        elif input_node_id_to_arg and inp.id in input_node_id_to_arg:
            # External input: function argument index [num_nodes, num_nodes + num_inputs)
            inputs.append(input_node_id_to_arg[inp.id])
        else:
            # Fallback: use -1 sentinel (should not happen with correct mapping)
            inputs.append(-1)
    return {
        "op": node.op_name,
        "inputs": inputs,
        "attrs": node.attrs,
        "shape": list(node.shape),
        "dtype": node.dtype,
    }


def serialize_subgraph(subgraph: Subgraph) -> str:
    """Serialize a Subgraph to JSON string for C++ consumption.

    The JSON structure:
    {
        "nodes": [
            {
                "op": "EWiseAdd",
                "inputs": [0, 1],
                "attrs": {},
                "shape": [128, 32, 32, 16],
                "dtype": "float32"
            },
            ...
        ],
        "inputs": [
            {"node_id": 5, "shape": [128, 32, 32, 3], "dtype": "float32"},
            ...
        ],
        "params": [
            {"name": "conv1.weight", "shape": [7, 7, 3, 16], "dtype": "float32"},
            ...
        ],
        "num_outputs": 2,
        "fused_groups": [[0, 3], [5, 7]],
        "output_node_indices": [0, 4],
        "mutation_map": {"bn1.running_mean": 1}
    }

    Returns:
        JSON string representation of the subgraph.
    """
    # Build local index mapping: global node_id -> local index in subgraph.nodes
    node_id_to_index: Dict[int, int] = {}
    for idx, node in enumerate(subgraph.nodes):
        node_id_to_index[node.id] = idx

    # Serialize nodes
    nodes_json = [_node_to_dict(node, node_id_to_index) for node in subgraph.nodes]

    # Serialize inputs
    inputs_json = []
    for node_id in subgraph.input_mapping:
        # We need shape/dtype info from the original graph context
        # This is filled in by the caller who has access to the full graph
        inputs_json.append({"node_id": node_id})

    # Serialize params
    params_json = []
    for param_name in subgraph.param_mapping:
        params_json.append({"name": param_name})

    # Collect output node indices
    output_node_indices = []
    for node_id in subgraph.output_mapping:
        if node_id in node_id_to_index:
            output_node_indices.append(node_id_to_index[node_id])

    # Build the full JSON structure
    serialized = {
        "nodes": nodes_json,
        "inputs": inputs_json,
        "params": params_json,
        "num_outputs": len(subgraph.output_mapping),
        "fused_groups": subgraph.fused_groups,
        "output_node_indices": output_node_indices,
        "mutation_map": subgraph.mutation_map,
    }

    return json.dumps(serialized)


def serialize_subgraph_full(
    subgraph: Subgraph,
    graph: Graph,
) -> str:
    """Serialize a Subgraph with full shape/dtype information resolved from the Graph context.

    This is the production version that fills in shape/dtype for inputs and params
    by looking up the nodes in the original graph.

    Args:
        subgraph: The subgraph to serialize
        graph: The complete computation graph (for shape/dtype resolution)

    Returns:
        JSON string representation of the subgraph with full shape/dtype info.
    """
    num_nodes = len(subgraph.nodes)

    # Build local index mapping: node_id → local index within subgraph
    node_id_to_index: Dict[int, int] = {}
    for idx, node in enumerate(subgraph.nodes):
        node_id_to_index[node.id] = idx

    # Build external input mapping: input node_id → function argument index
    # Function arguments are ordered: [input_shapes..., param_shapes...]
    # External input at position i in input_mapping → arg_index = num_nodes + i
    input_node_id_to_arg: Dict[int, int] = {}
    for arg_idx, node_id in enumerate(subgraph.input_mapping):
        input_node_id_to_arg[node_id] = num_nodes + arg_idx

    # Serialize nodes with correct external input indices
    nodes_json = [
        _node_to_dict(node, node_id_to_index, input_node_id_to_arg, num_nodes)
        for node in subgraph.nodes
    ]

    # Serialize inputs with shape/dtype from graph context
    inputs_json = []
    for node_id in subgraph.input_mapping:
        try:
            src_node = graph.find_node(node_id)
            inputs_json.append({
                "node_id": node_id,
                "shape": list(src_node.shape),
                "dtype": src_node.dtype,
            })
        except KeyError:
            # Node not in graph._node_index — use defaults
            inputs_json.append({
                "node_id": node_id,
                "shape": [],
                "dtype": "float32",
            })

    # Serialize params with shape/dtype from graph.params
    params_json = []
    for param_name in subgraph.param_mapping:
        if param_name in graph.params:
            pnode = graph.params[param_name]
            params_json.append({
                "name": param_name,
                "shape": list(pnode.shape),
                "dtype": pnode.dtype,
            })
        else:
            params_json.append({
                "name": param_name,
                "shape": [],
                "dtype": "float32",
            })

    # Collect output node indices
    output_node_indices = []
    for node_id in subgraph.output_mapping:
        if node_id in node_id_to_index:
            output_node_indices.append(node_id_to_index[node_id])

    # Build the full JSON structure
    serialized = {
        "nodes": nodes_json,
        "inputs": inputs_json,
        "params": params_json,
        "num_outputs": len(subgraph.output_mapping),
        "fused_groups": subgraph.fused_groups,
        "output_node_indices": output_node_indices,
        "mutation_map": subgraph.mutation_map,
    }

    return json.dumps(serialized)
