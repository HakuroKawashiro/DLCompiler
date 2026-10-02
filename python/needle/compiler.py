"""MLIR compiler entry point — jit() and CompiledModule.

This is the main user-facing API for MLIR compilation. Users call ndl.jit(model)
to enable JIT compilation of the model's forward pass.

When MLIR C++ backend is available:
  - First forward: trace -> partition -> compile -> JIT execute -> cache
  - Subsequent: JIT execute only (zero Python overhead for compute)

When MLIR C++ backend is NOT available:
  - Falls back to eager execution via self.original(*args)
  - Trace/partition still runs for testing the compiler pipeline
  - Autograd is preserved (backward() works correctly)
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

from .autograd import Tensor, TensorOp
from .nn.nn_basic import Module, Parameter
from .graph import Graph, GraphNode
from .tracer import Tracer
from .partition import Subgraph, partition_graph
from .serialize import serialize_subgraph_full

logger = logging.getLogger("needle.compiler")

# Check if MLIR C++ backend is actually functional
_MLIR_AVAILABLE = False
try:
    from .backend_mlir import MLIRDevice  # noqa: F401
    _MLIR_AVAILABLE = MLIRDevice().enabled()
except (ImportError, Exception):
    pass


def jit(model: Module) -> "CompiledModule":
    """Enable JIT compilation for a model.

    Wraps the model in a CompiledModule that automatically traces, partitions,
    compiles, and JIT-executes the forward pass on first invocation.

    Usage:
        model = ndl.jit(ResNet9())
        output = model(x)  # PyTorch-style: compilation is transparent
    """
    return CompiledModule(model)


class CompiledModule(Module):
    """Wrapper that JIT-compiles the forward pass via MLIR.

    Inherits nn.Module — transparent drop-in replacement.
    """

    def __init__(self, model: Module):
        super().__init__()
        self.original = model
        self.training = model.training

        # Compilation state
        self.graph: Optional[Graph] = None
        self.subgraphs: List[Subgraph] = []
        self.is_compiled: bool = False
        self._mode: Optional[bool] = None

        # Cached compiled function metadata
        self.compiled_fns: List[Tuple[str, List[int], Dict[int, int], Dict[str, int]]] = []

        # Parameter registry: param_name -> Parameter reference
        self.param_registry: Dict[str, Parameter] = {}

        # Run naming pass on original model
        self._name_modules(self.original)
        self._build_param_registry(self.original)

    # ── Module naming ──────────────────────────────────────────────────

    def _name_modules(self, module: Module, prefix: str = "") -> None:
        module._name = prefix if prefix else module.__class__.__name__
        for attr_name, value in module.__dict__.items():
            if attr_name.startswith('_'):
                continue
            if isinstance(value, Module):
                child_prefix = f"{prefix}.{attr_name}" if prefix else attr_name
                self._name_modules(value, child_prefix)
            elif isinstance(value, (list, tuple)):
                for idx, item in enumerate(value):
                    if isinstance(item, Module):
                        child_prefix = f"{prefix}.{attr_name}.{idx}" if prefix else f"{attr_name}.{idx}"
                        self._name_modules(item, child_prefix)

    def _build_param_registry(self, module: Module, prefix: str = "") -> None:
        for attr_name, value in module.__dict__.items():
            if attr_name.startswith('_'):
                continue
            param_prefix = f"{prefix}.{attr_name}" if prefix else attr_name
            if isinstance(value, Parameter):
                self.param_registry[param_prefix] = value
            elif isinstance(value, Module):
                self._build_param_registry(value, param_prefix)
            elif isinstance(value, (list, tuple)):
                for idx, item in enumerate(value):
                    item_prefix = f"{param_prefix}.{idx}"
                    if isinstance(item, Parameter):
                        self.param_registry[item_prefix] = item
                    elif isinstance(item, Module):
                        self._build_param_registry(item, item_prefix)

    # ── Core forward ───────────────────────────────────────────────────

    def forward(self, *args) -> Tensor:
        current_mode = self.original.training
        need_recompile = not self.is_compiled or self._mode != current_mode

        # Trace graph on first call (always, for partition inspection)
        if need_recompile or self.graph is None:
            self._trace_graph(current_mode, *args)

        # Execute: use MLIR JIT if available, otherwise eager fallback
        if _MLIR_AVAILABLE and self.is_compiled:
            return self._jit_forward(*args)
        else:
            return self._eager_forward(*args)

    def _trace_graph(self, current_mode: bool, *args) -> None:
        """Run the original model under Tracer to capture the computation graph.

        This populates self.graph, self.subgraphs, and self.compiled_fns.
        The actual execution result comes from self.original(*args).
        """
        logger.info("CompiledModule: tracing graph...")

        with Tracer() as tracer:
            output = self.original(*args)

        self.graph = tracer.graph

        # Mark output nodes
        if isinstance(output, tuple):
            for t in output:
                if id(t) in tracer.tensor_to_node:
                    self.graph.outputs.append(tracer.tensor_to_node[id(t)])
        else:
            if id(output) in tracer.tensor_to_node:
                self.graph.outputs.append(tracer.tensor_to_node[id(output)])

        self.graph.topo_sort()
        self.subgraphs = partition_graph(self.graph, tracer)

        # Generate cache keys for each subgraph
        self.compiled_fns = []
        for sg in self.subgraphs:
            subgraph_json = serialize_subgraph_full(sg, self.graph)
            cache_key = hashlib.sha256(subgraph_json.encode()).hexdigest()[:32]
            self.compiled_fns.append((
                cache_key,
                sg.input_mapping,
                sg.output_mapping,
                sg.mutation_map,
            ))

        # If MLIR is available, attempt compilation now
        if _MLIR_AVAILABLE:
            self._compile_subgraphs()

        self.is_compiled = _MLIR_AVAILABLE
        self._mode = current_mode

    def _compile_subgraphs(self) -> None:
        """Compile all subgraphs via MLIR and cache the JIT functions.

        This calls into the C++ needle_compiler module once per subgraph.
        The first forward after this will use the JIT functions.
        """
        from .backend_mlir import MLIRDevice
        mlir_device = MLIRDevice()

        # Build buffer pool for first execution
        buffer_pool: Dict[int, object] = {}
        for inp_node in self.graph.inputs:
            if inp_node.tensor is not None:
                buffer_pool[inp_node.id] = inp_node.tensor.realize_cached_data()
        for param_name, param_node in self.graph.params.items():
            if param_node.tensor is not None:
                buffer_pool[param_node.id] = param_node.tensor.realize_cached_data()

        for sg_idx, (cache_key, input_mapping, output_mapping, mutation_map) in enumerate(self.compiled_fns):
            sg = self.subgraphs[sg_idx]
            subgraph_json = serialize_subgraph_full(sg, self.graph)

            input_bufs = [buffer_pool.get(nid) for nid in input_mapping]
            import numpy as np
            output_bufs = []
            for node_id, out_idx in sorted(output_mapping.items(), key=lambda x: x[1]):
                for n in sg.nodes:
                    if n.id == node_id:
                        output_bufs.append(np.empty(n.shape, dtype=np.float32))
                        break

            try:
                outputs = mlir_device.compile_and_run(
                    cache_key, subgraph_json, input_bufs, output_bufs)
                for (node_id, out_idx), out_buf in zip(
                    sorted(output_mapping.items(), key=lambda x: x[1]),
                    outputs if outputs else output_bufs,
                ):
                    buffer_pool[node_id] = out_buf
                    try:
                        node = self.graph.find_node(node_id)
                        if node.tensor is not None:
                            node.tensor.cached_data = out_buf
                    except KeyError:
                        pass
            except Exception as e:
                logger.warning(f"Subgraph {sg_idx} compilation failed: {e}")

        self._write_back_mutations(buffer_pool)

    def _jit_forward(self, *args) -> Tensor:
        """Execute via cached JIT functions (MLIR available)."""
        buffer_pool: Dict[int, object] = {}

        # Map new args to graph input nodes
        for i, inp_node in enumerate(self.graph.inputs):
            if i < len(args):
                buffer_pool[inp_node.id] = args[i].realize_cached_data()

        # Refresh param buffers
        for param_name, param_node in self.graph.params.items():
            if param_name in self.param_registry:
                buffer_pool[param_node.id] = self.param_registry[param_name].realize_cached_data()

        from .backend_mlir import MLIRDevice
        mlir_device = MLIRDevice()

        for sg_idx, (cache_key, input_mapping, output_mapping, mutation_map) in enumerate(self.compiled_fns):
            sg = self.subgraphs[sg_idx]
            input_bufs = [buffer_pool.get(nid) for nid in input_mapping]
            import numpy as np
            output_bufs = []
            for node_id, out_idx in sorted(output_mapping.items(), key=lambda x: x[1]):
                for n in sg.nodes:
                    if n.id == node_id:
                        output_bufs.append(np.empty(n.shape, dtype=np.float32))
                        break

            outputs = mlir_device.compile_and_run(
                cache_key, "", input_bufs, output_bufs)

            for (node_id, out_idx), out_buf in zip(
                sorted(output_mapping.items(), key=lambda x: x[1]),
                outputs if outputs else output_bufs,
            ):
                buffer_pool[node_id] = out_buf
                try:
                    node = self.graph.find_node(node_id)
                    if node.tensor is not None:
                        node.tensor.cached_data = out_buf
                except KeyError:
                    pass

        self._write_back_mutations(buffer_pool)

        if self.graph.outputs:
            return self.graph.outputs[0].tensor

        return args[0] if args else None

    def _eager_forward(self, *args) -> Tensor:
        """Execute via standard eager path (real autograd, no JIT).

        When MLIR is not available, this is the default execution path.
        It runs self.original(*args) directly, which goes through all
        TensorOp calls with proper autograd tracking.

        The graph was already traced in _trace_graph for inspection.
        """
        return self.original(*args)

    # ── Parameter buffer management ────────────────────────────────────

    def _refresh_param_buffers(self) -> Dict[str, object]:
        bufs: Dict[str, object] = {}
        for param_name, param in self.param_registry.items():
            bufs[param_name] = param.realize_cached_data()
        return bufs

    def _write_back_mutations(self, buffer_pool: Dict[int, object]) -> None:
        for sg in self.subgraphs:
            for param_name, output_idx in sg.mutation_map.items():
                if param_name in self.param_registry:
                    param = self.param_registry[param_name]
                    for node_id, out_idx in sg.output_mapping.items():
                        if out_idx == output_idx and node_id in buffer_pool:
                            param.data = Tensor(buffer_pool[node_id])
                            break

    # ── Mode management ────────────────────────────────────────────────

    def train(self) -> None:
        was_training = self.original.training
        self.original.train()
        self.training = True
        if not was_training and self.is_compiled:
            self.is_compiled = False

    def eval(self) -> None:
        was_training = self.original.training
        self.original.eval()
        self.training = False
        if was_training and self.is_compiled:
            self.is_compiled = False

    # ── Parameter & serialization delegation ───────────────────────────

    def parameters(self) -> List[Parameter]:
        return self.original.parameters()

    def state_dict(self) -> Dict[str, object]:
        if hasattr(self.original, 'state_dict'):
            return self.original.state_dict()
        from .serialization import _state_dict
        return _state_dict(self.original)

    def load_state_dict(self, d: Dict[str, object]) -> None:
        if hasattr(self.original, 'load_state_dict'):
            self.original.load_state_dict(d)
        else:
            from .serialization import _load_state_dict
            _load_state_dict(self.original, d)
        self.is_compiled = False
