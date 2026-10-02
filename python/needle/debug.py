"""Debug mode management for MLIR compiler.

When debug mode is active, all tensor operations go through NumPy eager execution
instead of MLIR JIT, allowing correctness comparison between the two paths.
"""

_debug_mode: bool = False


def debug() -> None:
    """Enable NumPy debug mode.

    In debug mode, all tensor operations execute via NumPy (eager, no JIT).
    Used for verifying MLIR compilation correctness by comparing outputs.

    Note: This triggers a re-import of backend_selection, which re-reads
    the _debug_mode flag to select the NumPy backend.

    Usage:
        ndl.debug()
        # ... run model ...  (all ops go through NumPy)
        ndl.debug()  # toggle back
    """
    global _debug_mode
    _debug_mode = True

    # Re-import backend selection to apply the change
    from . import backend_selection
    # Trigger re-resolve
    from .backend_selection import _resolve_backend
    backend_selection.array_api, backend_selection.NDArray, \
        backend_selection.all_devices, backend_selection.cpu, \
        backend_selection.default_device, backend_selection.Device = _resolve_backend()

    # Update autograd's imports
    from . import autograd
    autograd.array_api = backend_selection.array_api
    autograd.NDArray = backend_selection.NDArray


def is_debug() -> bool:
    """Return whether debug mode is currently active."""
    return _debug_mode


def _set_debug(val: bool) -> None:
    """Internal setter for the debug flag."""
    global _debug_mode
    _debug_mode = val
