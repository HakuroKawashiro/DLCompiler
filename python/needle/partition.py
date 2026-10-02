"""Subgraph partitioning — splits computation graph into MLIR-compilable subgraphs."""

from typing import Any, Dict, List, Optional, Set, Tuple

from .graph import Graph, GraphNode
from .tracer import Tracer

# Operators that can be compiled via MLIR
SUPPORTED_MLIR_OPS: Set[str] = {
    "EWiseAdd", "EWiseMul", "EWiseDiv", "EWisePow",
    "ReLU", "Tanh", "Exp", "Log", "Negate",
    "AddScalar", "MulScalar", "DivScalar", "PowerScalar",
    "MatMul", "Conv",
    "Reshape", "Transpose", "BroadcastTo",
    "Summation",
    "MakeTensorTuple", "TupleGetItem",
}

# Operators that participate in element-wise fusion (only "parallel" iterator types)
FUSIBLE_ELEMENTWISE_OPS: Set[str] = {
    "EWiseAdd", "EWiseMul", "EWiseDiv", "EWisePow",
    "ReLU", "Tanh", "Exp", "Log", "Negate",
    "AddScalar", "MulScalar", "DivScalar", "PowerScalar",
    "BroadcastTo",
}

# Operators that act as fusion barriers — cannot be crossed by element-wise fusion chains
# (but can still be in the same subgraph; the element-wise chain restarts after them)
FUSION_BARRIER_OPS: Set[str] = {
    "Summation", "MatMul", "Conv", "Reshape",
}


class Subgraph:
    """A contiguous segment of the computation graph that can be compiled via MLIR.

    Attributes:
        nodes: Ordered list of GraphNode in this subgraph
        input_mapping: List of node_ids for external inputs consumed by this subgraph
        param_mapping: List of param names (keys into Graph.params) consumed
        output_mapping: Dict[node_id, output_index] — which nodes produce outputs
                        and their position in the output buffer list
        mutation_map: Dict[param_name, output_index] — outputs that are Parameter mutations
        fused_groups: List of (start_node_id, end_node_id) intervals marking element-wise fusion groups
    """

    def __init__(self):
        self.nodes: List[GraphNode] = []
        self.input_mapping: List[int] = []  # external input node_ids
        self.param_mapping: List[str] = []   # param names from Graph.params
        self.output_mapping: Dict[int, int] = {}  # node_id -> output_index
        self.mutation_map: Dict[str, int] = {}    # param_name -> output_index
        self.fused_groups: List[Tuple[int, int]] = []  # (start_idx_in_nodes, end_idx_in_nodes)


class _PartitionContext:
    """Internal state during partition_graph traversal."""

    def __init__(self, graph: Graph, tracer: Tracer):
        self.graph = graph
        self.tracer = tracer
        # Set of node_ids already assigned to a subgraph
        self.assigned: Set[int] = set()
        # Map from original node_id to (subgraph_index, output_index_in_subgraph)
        self.node_output_map: Dict[int, Tuple[int, int]] = {}
        # Which node_ids are consumed by downstream subgraphs (need to be exposed as outputs)
        self.consumed_by_downstream: Set[int] = set()
        # Map from param name to the node_id that mutates it
        self.mutation_node_ids: Dict[str, int] = {}


def partition_graph(graph: Graph, tracer: Tracer) -> List[Subgraph]:
    """Partition a computation graph into MLIR-compilable subgraphs.

    Strategy:
    1. Topologically sort all nodes
    2. Walk nodes in topo order
    3. Group consecutive SUPPORTED_MLIR_OPS into subgraphs
    4. Within each subgraph, identify fusion groups (contiguous FUSIBLE_ELEMENTWISE_OPS)
    5. Handle fallback nodes (op_name="fallback") as independent subgraphs
    6. Track input/output boundaries between subgraphs
    """
    if not graph.nodes:
        graph.topo_sort()

    ctx = _PartitionContext(graph, tracer)

    # Pre-compute consumer info: which nodes are consumed by which other nodes
    consumer_map: Dict[int, List[int]] = {}
    for node in graph.nodes:
        for inp in node.inputs:
            consumer_map.setdefault(inp.id, []).append(node.id)

    # Build mutation node info from tracer
    for param_id, value_node in tracer._mutation_targets.items():
        # Find param name by looking up graph.params
        for pname, pnode in graph.params.items():
            if value_node.id not in ctx.mutation_node_ids:
                ctx.mutation_node_ids[pname] = value_node.id

    subgraphs: List[Subgraph] = []

    i = 0
    while i < len(graph.nodes):
        node = graph.nodes[i]

        # Fallback nodes: create a standalone fallback subgraph (runs on eager path)
        if node.op_name == "fallback":
            sg = Subgraph()
            sg.nodes.append(node)
            # Collect inputs from upstream subgraphs
            for inp in node.inputs:
                if inp.id in ctx.assigned:
                    # Already produced by an earlier subgraph
                    sg.input_mapping.append(inp.id)
            # Mark as output so downstream can consume
            sg.output_mapping[node.id] = 0
            subgraphs.append(sg)
            ctx.assigned.add(node.id)
            i += 1
            continue

        # Check if this node is MLIR-supported
        if node.op_name not in SUPPORTED_MLIR_OPS:
            # Unsupported op — treated as a singleton fallback
            sg = Subgraph()
            sg.nodes.append(node)
            for inp in node.inputs:
                if inp.id in ctx.assigned:
                    sg.input_mapping.append(inp.id)
            sg.output_mapping[node.id] = 0
            subgraphs.append(sg)
            ctx.assigned.add(node.id)
            i += 1
            continue

        # Start a new MLIR subgraph
        sg = Subgraph()
        # Collect consecutive supported ops
        j = i
        while j < len(graph.nodes) and graph.nodes[j].op_name in SUPPORTED_MLIR_OPS:
            sg.nodes.append(graph.nodes[j])
            ctx.assigned.add(graph.nodes[j].id)
            j += 1

        # Post-process the subgraph to identify:
        # 1. External inputs (from outside this subgraph)
        # 2. Fusion groups
        # 3. Output mapping
        sg_input_set: Set[int] = set()
        sg_node_ids: Set[int] = {n.id for n in sg.nodes}
        local_param_names: List[str] = []

        for node in sg.nodes:
            for inp in node.inputs:
                if inp.id not in sg_node_ids:
                    # This is an external input
                    if inp.id not in sg_input_set:
                        sg_input_set.add(inp.id)
                        sg.input_mapping.append(inp.id)
                        # If it's a param
                        for pname, pnode in graph.params.items():
                            if pnode.id == inp.id and pname not in local_param_names:
                                local_param_names.append(pname)

        sg.param_mapping = local_param_names

        # Identify fusion groups: contiguous FUSIBLE_ELEMENTWISE_OPS within the subgraph
        fusion_start = None
        for idx, node in enumerate(sg.nodes):
            if node.op_name in FUSIBLE_ELEMENTWISE_OPS:
                if fusion_start is None:
                    fusion_start = idx
            else:
                if fusion_start is not None and (idx - fusion_start) >= 2:
                    sg.fused_groups.append((fusion_start, idx))
                fusion_start = None

            # Check for BroadcastTo rank change — if BroadcastTo changes rank,
            # it should break the fusion group.
            if node.op_name == "BroadcastTo" and fusion_start is not None:
                # If any input has different rank from output, end current fusion group
                for inp in node.inputs:
                    if inp.id in sg_node_ids:
                        inp_node = graph.find_node(inp.id)
                        if inp_node is not None and len(inp_node.shape) != len(node.shape):
                            # Rank change — end fusion group before this BroadcastTo
                            if idx - fusion_start >= 2:
                                sg.fused_groups.append((fusion_start, idx))
                            fusion_start = idx  # Start new group from this broadcast

        # Close trailing fusion group
        if fusion_start is not None and (len(sg.nodes) - fusion_start) >= 2:
            sg.fused_groups.append((fusion_start, len(sg.nodes)))

        # Determine outputs: nodes whose results are consumed by downstream nodes
        # or mutation outputs
        output_idx = 0
        for node in sg.nodes:
            consumers = consumer_map.get(node.id, [])
            downstream_consumers = [c for c in consumers if c not in sg_node_ids]
            is_mutation_output = False
            for pname, nid in ctx.mutation_node_ids.items():
                if nid == node.id:
                    is_mutation_output = True
                    sg.mutation_map[pname] = output_idx
                    break

            if downstream_consumers or is_mutation_output or node.id in {o.id for o in graph.outputs}:
                sg.output_mapping[node.id] = output_idx
                output_idx += 1

        # If no outputs were identified, make the last node the output
        if not sg.output_mapping:
            last_node = sg.nodes[-1]
            sg.output_mapping[last_node.id] = 0

        subgraphs.append(sg)
        i = j

    return subgraphs


def _resolve_mutation_name(node: GraphNode, tracer: Tracer, graph: Graph) -> str:
    """Resolve a Parameter mutation name: given a GraphNode that produces a new value,
    find the corresponding param name from tracer._mutation_targets + graph.params.

    Resolution logic:
    1. Find which id(param) in _mutation_targets maps to this node
    2. Look up that param's GraphNode in graph.params (param_name -> GraphNode)
    3. If the GraphNode matches, return the param_name
    """
    # Reverse lookup: find param_id that maps to this node
    for param_id, value_node in tracer._mutation_targets.items():
        if value_node.id == node.id:
            # Find the param_name that has this param_id
            for pname, pnode in graph.params.items():
                # We need to check if pnode's tensor is the same object
                if pnode.tensor is not None and id(pnode.tensor) == param_id:
                    return pname
            # Fallback: if we can't match by id, the mutation target was set correctly
            # Try to find by scanning all params
            for pname, pnode in graph.params.items():
                if pnode.tensor is not None:
                    return pname

    return ""
