"""Runtime tracing for MLIR compiler — captures operator calls into a computation graph."""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from .graph import Graph, GraphNode
from .autograd import Tensor, TensorOp
from .nn.nn_basic import Parameter

logger = logging.getLogger("needle.tracer")


class Tracer:
    """Context manager that records all TensorOp calls into a Graph.

    Usage:
        with Tracer() as tracer:
            output = model(x)
        graph = tracer.graph
    """

    # Global active tracer reference (set on __enter__, cleared on __exit__)
    _ACTIVE_TRACER: Optional["Tracer"] = None

    def __init__(self):
        self.graph: Graph = Graph()
        # Map id(tensor) -> GraphNode for Tensor-to-GraphNode lookup
        self.tensor_to_node: Dict[int, GraphNode] = {}
        # Map id(Parameter) -> GraphNode that produces the new value for mutation write-back
        self._mutation_targets: Dict[int, GraphNode] = {}
        # Map id(Parameter) -> full parameter name (e.g. "conv1.weight")
        self._param_names: Dict[int, str] = {}
        # Stack for parameter naming during trace (module attribute path)
        self._param_name_stack: List[str] = []
        # Side effect ops registry (reserved for future inplace ops)
        self._side_effect_ops: Set[str] = set()

    def __enter__(self) -> "Tracer":
        Tracer._ACTIVE_TRACER = self
        # Set global tracing flag and active tracer in autograd module
        import needle.autograd as ag
        ag._TRACING = True
        ag._ACTIVE_TRACER = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        Tracer._ACTIVE_TRACER = None
        import needle.autograd as ag
        ag._TRACING = False
        ag._ACTIVE_TRACER = None
        return False  # Don't suppress exceptions

    def record(
        self,
        op: TensorOp,
        inputs: List[Tensor],
        output: Tensor,
    ) -> GraphNode:
        """Core recording function called by TensorOp.__call__ when tracing.

        Steps:
        1. Look up or create GraphNodes for each input Tensor
        2. Extract op name and attributes from the TensorOp instance
        3. Create a new GraphNode and add to the graph
        4. Map the output Tensor to the new GraphNode
        5. If output is a Parameter mutation, record in _mutation_targets

        Returns the created GraphNode.
        """
        # Step 1: Resolve input nodes
        input_nodes: List[GraphNode] = []
        for inp in inputs:
            node = self.get_or_create_input_node(inp)
            input_nodes.append(node)

        # Step 2: Extract op info
        op_name = self.infer_op_name(op)
        attrs = self.infer_attrs(op)

        # Step 3: Get output shape/dtype from the realized tensor
        output_np = output.realize_cached_data()
        shape = tuple(output_np.shape)
        dtype = str(output_np.dtype)

        # Step 4: Create graph node
        node = self.graph.add_node(op_name, input_nodes, attrs, shape, dtype)
        node.tensor = output

        # Step 5: Map output tensor to node
        self.tensor_to_node[id(output)] = node

        return node

    def infer_op_name(self, op: TensorOp) -> str:
        """Infer operator name string from TensorOp class name."""
        return op.__class__.__name__

    def infer_attrs(self, op: TensorOp) -> Dict[str, Any]:
        """Extract operator attributes (stride, padding, axes, etc.) from the op instance."""
        attrs: Dict[str, Any] = {}

        # Common attributes across ops
        for attr_name in ["stride", "padding", "axes", "axis", "scalar",
                           "momentum", "eps", "shape", "dilation"]:
            if hasattr(op, attr_name):
                val = getattr(op, attr_name)
                if val is not None:
                    attrs[attr_name] = val

        return attrs

    def get_or_create_input_node(self, tensor: Tensor) -> GraphNode:
        """Look up or create a GraphNode for a Tensor.

        - If tensor already has a mapping in tensor_to_node, return it.
        - If tensor is a Parameter, create a "param" placeholder and add to graph.params.
        - Otherwise, create an "input" placeholder and add to graph.inputs.
        """
        tid = id(tensor)
        if tid in self.tensor_to_node:
            return self.tensor_to_node[tid]

        # Determine shape/dtype
        np_data = tensor.realize_cached_data()
        shape = tuple(np_data.shape)
        dtype = str(np_data.dtype)

        if isinstance(tensor, Parameter):
            # Determine param name from the current stack context
            param_name = self._resolve_param_name(tensor)
            node = self.graph.add_node("param", [], {}, shape, dtype)
            node.tensor = tensor
            self.graph.params[param_name] = node
        else:
            node = self.graph.add_node("input", [], {}, shape, dtype)
            node.tensor = tensor
            self.graph.inputs.append(node)

        self.tensor_to_node[tid] = node
        return node

    def _resolve_param_name(self, tensor: Parameter) -> str:
        """Resolve the full parameter name for a Parameter tensor.

        First checks if the Parameter was pre-registered by Module.__call__
        (via _param_names). Otherwise, falls back to stack-based naming.
        """
        tid = id(tensor)

        # Check pre-registered names (set by Module.__call__ during trace)
        if tid in self._param_names:
            return self._param_names[tid]

        # Fallback: use the _param_name_stack to construct a name
        if self._param_name_stack:
            prefix = ".".join(self._param_name_stack)
            return f"{prefix}._param_{tid}"
        else:
            return f"_param_{tid}"

    def pause(self) -> None:
        """Temporarily disable tracing (e.g. for Dropout random mask generation)."""
        import needle.autograd as ag
        ag._TRACING = False

    def resume(self) -> None:
        """Resume tracing after a pause()."""
        import needle.autograd as ag
        ag._TRACING = True

    def mark_fallback(self, tensor: Tensor) -> GraphNode:
        """Mark a tensor as requiring fallback (eager) execution.

        Creates a placeholder node with op_name="fallback" that partition_graph
        recognizes and handles by creating an independent fallback subgraph.
        """
        np_data = tensor.realize_cached_data()
        shape = tuple(np_data.shape)
        dtype = str(np_data.dtype)

        # Find inputs to the fallback node from the tensor_to_node mapping
        # (the tensor was produced by some ops, we track its producing node)
        node = self.graph.add_node("fallback", [], {}, shape, dtype)
        node.tensor = tensor
        self.tensor_to_node[id(tensor)] = node
        return node

    def record_mutation(self, param: Parameter, value_node: GraphNode) -> None:
        """Record that a Parameter's .data was set during trace,
        mapping the Parameter to the GraphNode that produces the new value.

        Called from Tensor.data setter when _TRACING is True.
        """
        pid = id(param)
        # If there's already a mapping for this param, the new one overwrites
        self._mutation_targets[pid] = value_node
