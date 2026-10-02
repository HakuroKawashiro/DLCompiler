"""MLIR backend — provides MLIRDevice wrapping the C++ needle_compiler pybind11 module.

The C++ side (src/compiler/*.cc) implements:
- MLIR IR generation (linalg.conv_2d, linalg.matmul, linalg.generic, etc.)
- Pass pipeline (fusion, tiling, vectorization, lowering to LLVM IR)
- LLVM ORC JIT compilation

The Python side provides:
- MLIRDevice.compile_and_run(): main entry point for JIT compilation and execution
- Graceful fallback when the C++ module is not available
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("needle.backend_mlir")

# Try to import the C++ pybind11 module
_compiler = None
try:
    from . import needle_compiler as _compiler
except ImportError:
    logger.warning(
        "MLIR compiler (needle_compiler) not available. "
        "JIT compilation disabled — models will run via eager NDArray fallback. "
        "Build the MLIR compiler with: make mlir"
    )


class MLIRDevice:
    """Wrapper around the C++ MLIR compiler.

    Provides the Python-facing interface for JIT compilation and execution.
    The actual MLIR/LLVM heavy lifting happens in C++ via pybind11.
    """

    def __init__(self):
        self.mod = _compiler
        self._enabled = self.mod is not None

    def enabled(self) -> bool:
        """Check whether the MLIR compiler module is available."""
        return self._enabled

    def compile_and_run(
        self,
        cache_key: str,
        subgraph_json: str,
        input_buffers: List[Any],
        output_buffers: List[Any],
    ) -> List[Any]:
        """Compile a subgraph to MLIR, JIT compile, and execute.

        Args:
            cache_key: SHA256 hash for JIT cache lookup (32 hex chars = 128 bits)
            subgraph_json: JSON serialization of the subgraph (empty if cached)
            input_buffers: List of numpy arrays or NDArrays for inputs
            output_buffers: List of pre-allocated numpy arrays for outputs
                            (JIT writes results directly into these buffers)

        Returns:
            List of output buffers (same objects passed in, now filled with results)

        Raises:
            RuntimeError: If the C++ module is not available
        """
        if not self._enabled:
            raise RuntimeError("MLIR compiler not available")

        # Convert all buffers to numpy arrays for pybind11
        import numpy as np

        np_inputs = []
        for buf in input_buffers:
            if hasattr(buf, 'numpy'):
                np_inputs.append(np.ascontiguousarray(buf.numpy().astype(np.float32)))
            elif isinstance(buf, np.ndarray):
                np_inputs.append(np.ascontiguousarray(buf.astype(np.float32)))
            else:
                np_inputs.append(np.ascontiguousarray(np.array(buf, dtype=np.float32)))

        np_outputs = []
        for buf in output_buffers:
            if hasattr(buf, 'numpy'):
                np_outputs.append(np.ascontiguousarray(buf.numpy().astype(np.float32)))
            elif isinstance(buf, np.ndarray):
                np_outputs.append(np.ascontiguousarray(buf.astype(np.float32)))
            else:
                np_outputs.append(np.empty(1, dtype=np.float32))

        # Call C++ compile_and_run
        result = self.mod.compile_and_run(
            cache_key,
            subgraph_json,
            np_inputs,
            np_outputs,
        )

        return list(result)
