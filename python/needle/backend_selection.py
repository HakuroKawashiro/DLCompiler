"""Logic for backend selection"""
import os


BACKEND = os.environ.get("NEEDLE_BACKEND", "nd")


def _resolve_backend():
    """Resolve the backend based on global debug mode and environment.

    - Debug mode (ndl.debug()): NumPy backend for correctness verification
    - Default mode: NDArray backend (C++ pybind11) for eager execution,
      MLIR JIT for compiled subgraphs (via CompiledModule)
    """
    from .debug import _debug_mode

    if _debug_mode:
        # Debug mode: NumPy eager execution for correctness comparison
        import numpy as debug_array_api

        class DebugDevice:
            def __init__(self, name, mod):
                self.name = name
                self.mod = mod
            def __eq__(self, other):
                return hasattr(other, 'name') and self.name == other.name
            def __repr__(self):
                return self.name + "()"
            def __hash__(self):
                return hash(self.name)
            def enabled(self):
                return True

        def _debug_cpu():
            return DebugDevice("cpu", None)

        def _debug_cpu_numpy():
            return DebugDevice("cpu_numpy", None)

        def _debug_all_devices():
            return [_debug_cpu(), _debug_cpu_numpy()]

        array_api = debug_array_api
        NDArray = debug_array_api.ndarray
        all_devices = _debug_all_devices
        cpu = _debug_cpu
        default_device = _debug_cpu
        Device = DebugDevice
        return array_api, NDArray, all_devices, cpu, default_device, Device

    # Default mode
    if BACKEND == "nd":
        from . import backend_ndarray as array_api
        from .backend_ndarray import (
            all_devices,
            cuda,
            cpu,
            cpu_numpy,
            default_device,
            BackendDevice as Device,
        )
        NDArray = array_api.NDArray
    elif BACKEND == "np":
        import numpy as array_api
        from .backend_numpy import all_devices, cpu, default_device, Device
        NDArray = array_api.ndarray
    else:
        raise RuntimeError("Unknown needle array backend %s" % BACKEND)

    return array_api, NDArray, all_devices, cpu, default_device, Device


array_api, NDArray, all_devices, cpu, default_device, Device = _resolve_backend()
