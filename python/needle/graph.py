"""Computation graph data structures for MLIR compiler."""

import itertools
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Set

# Global node ID counter
_NODE_ID_COUNTER = itertools.count()


class GraphNode:
    """Represents a single operator node in the computation graph.

    Attributes:
        op_name: Operator name string (e.g. "EWiseAdd", "MatMul", "Conv")
        inputs: List of input GraphNode references
        attrs: Operator-specific attributes (e.g. stride, padding, axes)
        shape: Output tensor shape
        dtype: Output tensor dtype (e.g. "float32")
        id: Unique auto-incrementing node ID
        tensor: Reference to the original Tensor object (for backward/gradient lookups)
    """

    def __init__(
        self,
        op_name: str,
        inputs: List["GraphNode"],
        attrs: Dict[str, Any],
        shape: tuple,
        dtype: str = "float32",
    ):
        self.op_name = op_name
        self.inputs = inputs
        self.attrs = attrs
        self.shape = shape
        self.dtype = dtype
        self.id = next(_NODE_ID_COUNTER)
        self.tensor = None  # Set by Tracer.record()


class Graph:
    """Container for a computation graph DAG.

    Attributes:
        name: Optional graph name for debugging
        inputs: External input nodes (not parameters)
        params: Parameter nodes keyed by param name (e.g. "conv1.weight")
        outputs: Output node(s) of the graph
        nodes: Topologically sorted list of all nodes
        _node_index: O(1) lookup dict from node_id -> GraphNode
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.inputs: List[GraphNode] = []
        self.params: Dict[str, GraphNode] = {}
        self.outputs: List[GraphNode] = []
        self.nodes: List[GraphNode] = []
        self._node_index: Dict[int, GraphNode] = {}

    def add_node(
        self,
        op_name: str,
        inputs: List[GraphNode],
        attrs: Dict[str, Any],
        shape: tuple,
        dtype: str = "float32",
    ) -> GraphNode:
        """Create a new node and add to internal list with auto-increment ID."""
        node = GraphNode(op_name, inputs, attrs, shape, dtype)
        self._node_index[node.id] = node
        return node

    def find_node(self, node_id: int) -> GraphNode:
        """O(1) lookup of a node by its ID."""
        if node_id not in self._node_index:
            raise KeyError(f"GraphNode with id {node_id} not found in graph")
        return self._node_index[node_id]

    def topo_sort(self) -> None:
        """Kahn's algorithm topological sort of all nodes reachable from outputs.

        Result is stored in self.nodes. After sorting, self.nodes[i]'s
        dependencies are guaranteed to be in self.nodes[:i].
        """
        # Collect all reachable nodes from outputs
        reachable: Set[int] = set()
        visited: Set[int] = set()

        def dfs(n: GraphNode):
            if n.id in visited:
                return
            visited.add(n.id)
            reachable.add(n.id)
            for inp in n.inputs:
                dfs(inp)

        for out_node in self.outputs:
            dfs(out_node)

        # Also collect from params and inputs
        for p_node in self.params.values():
            dfs(p_node)
        for inp_node in self.inputs:
            dfs(inp_node)

        # Build adjacency and in-degree
        in_degree: Dict[int, int] = {}
        adjacency: Dict[int, List[GraphNode]] = {}

        reachable_nodes = {nid: self._node_index[nid] for nid in reachable}

        for nid, node in reachable_nodes.items():
            in_degree.setdefault(nid, 0)
            for inp in node.inputs:
                if inp.id in reachable:
                    in_degree[nid] = in_degree.get(nid, 0) + 1

        for nid, node in reachable_nodes.items():
            for inp in node.inputs:
                if inp.id in reachable:
                    adjacency.setdefault(inp.id, []).append(node)

        # Kahn's algorithm
        queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
        sorted_nodes: List[GraphNode] = []

        while queue:
            nid = queue.popleft()
            node = reachable_nodes[nid]
            sorted_nodes.append(node)
            for consumer in adjacency.get(nid, []):
                in_degree[consumer.id] -= 1
                if in_degree[consumer.id] == 0:
                    queue.append(consumer.id)

        # If there are cycles, include remaining nodes anyway
        remaining = [n for nid, n in reachable_nodes.items() if nid not in {x.id for x in sorted_nodes}]
        sorted_nodes.extend(remaining)

        self.nodes = sorted_nodes

    def find_eligible_group(self, node: GraphNode) -> List[GraphNode]:
        """Backward DFS from `node` collecting all fusible element-wise nodes.

        Fusible ops are those whose MLIR linalg.generic has only "parallel"
        iterator types. Stops at non-fusible nodes (MatMul, Conv, Summation,
        Reshape, etc.).

        Note: This function is a helper for partition_graph to identify
        contiguous element-wise chains. The actual fusible set is defined
        in partition.py (FUSIBLE_ELEMENTWISE_OPS).
        """
        from .partition import FUSIBLE_ELEMENTWISE_OPS, FUSION_BARRIER_OPS

        group: List[GraphNode] = []
        visited: Set[int] = set()

        def dfs(n: GraphNode):
            if n.id in visited:
                return
            visited.add(n.id)

            if n.op_name not in FUSIBLE_ELEMENTWISE_OPS:
                # Stop at non-fusible ops (barrier or unsupported)
                return

            # Recurse into inputs first (bottom-up)
            for inp in n.inputs:
                dfs(inp)

            group.append(n)

        dfs(node)
        return group
