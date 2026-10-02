"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def to(self, device) -> None:
        """Move all parameters to the specified device (in-place)."""
        for param in self.parameters():
            param.data = param.to(device).data
        for child in self._children():
            child.to(device)

    def __call__(self, *args, **kwargs):
        # MLIR tracing: maintain parameter name stack for Parameter naming
        import needle.autograd as ag
        if ag._TRACING and ag._ACTIVE_TRACER is not None:
            tracer = ag._ACTIVE_TRACER
            module_name = getattr(self, '_name', None)

            if module_name is not None:
                tracer._param_name_stack.append(module_name)

            # Pre-register all Parameters in this module with their full names
            # so that when they appear as inputs during trace, they get correct names.
            from .nn_basic import Parameter as ParamType
            for attr_name, value in self.__dict__.items():
                if isinstance(value, ParamType):
                    full_name = f"{module_name}.{attr_name}" if module_name else attr_name
                    # Pre-populate the parameter name mapping
                    tracer._param_names[id(value)] = full_name
                    # Also register the param input node early
                    if id(value) not in tracer.tensor_to_node:
                        tracer.get_or_create_input_node(value)

            result = self.forward(*args, **kwargs)

            if module_name is not None:
                tracer._param_name_stack.pop()

            return result
        else:
            return self.forward(*args, **kwargs)

    def state_dict(self) -> dict:
        """Return a dictionary mapping parameter names to numpy arrays.

        Uses the same traversal as _name_modules to assign hierarchical names
        (e.g., "conv1.weight").
        """
        return _build_state_dict(self)

    def load_state_dict(self, d: dict) -> None:
        """Load parameters from a state_dict.

        Args:
            d: Dictionary mapping parameter names to numpy arrays.

        Raises:
            ValueError: If a parameter shape doesn't match.
        """
        _restore_state_dict(self, d)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features), device=device, dtype=dtype)
        if bias:
            self.bias = Parameter(ops.transpose(init.kaiming_uniform(out_features, 1)), device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = X @ self.weight
        
        if hasattr(self, 'bias'):
            y = y + ops.broadcast_to(self.bias, y.shape)
        
        return y
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        X_pru = 1
        for i in range(1, len(X.shape)):
            X_pru *= X.shape[i]
        y = ops.reshape(X, (X.shape[0], X_pru))
        return y
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
            
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y_one_hot = init.one_hot(logits.shape[1], y, device=logits.device)
        v1 = ops.logsumexp(logits, axes = 1) - ops.summation(logits * y_one_hot, axes = 1)
        loss = ops.summation(v1) / v1.shape[0]
        return loss
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim), device=device, dtype=dtype)
        self.bias = Parameter(init.zeros(dim), device=device, dtype=dtype)
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        shape_select = []
        cnt = 1
        for i in range(len(x.shape) - 1):
            shape_select.append(i)
            cnt *= x.shape[i]
        shape_select = tuple(shape_select)
        
        if self.training:
            E_x = ops.summation(x, shape_select) / cnt
            x_minus_E_x = x - ops.broadcast_to(E_x, x.shape)
            Var_x = ops.summation(x_minus_E_x ** 2, shape_select) / cnt
            y = ops.broadcast_to(self.weight, x.shape) * (x_minus_E_x / ops.broadcast_to((Var_x + self.eps) ** 0.5, x.shape))
            y = y + ops.broadcast_to(self.bias, x.shape)
            self.running_mean.data = (1 - self.momentum) * self.running_mean.data + self.momentum * E_x.data
            self.running_var.data = (1 - self.momentum) * self.running_var.data + self.momentum * Var_x.data
        else:
            x_minus_mu = x - ops.broadcast_to(self.running_mean, x.shape)
            y = ops.broadcast_to(self.weight, x.shape) * (x_minus_mu / ops.broadcast_to((self.running_var + self.eps) ** 0.5, x.shape))
            y = y + ops.broadcast_to(self.bias, x.shape)
        
        return y
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim), device=device, dtype=dtype)
        self.bias = Parameter(init.zeros(dim), device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        shape_ex = list(x.shape)
        shape_ex[-1] = 1
        shape_ex = tuple(shape_ex)
        E_x = ops.broadcast_to(ops.reshape(ops.summation(x, -1) / x.shape[-1], shape_ex), x.shape)
        x_minus_E_x = x - E_x
        Var_x = ops.reshape(ops.summation(x_minus_E_x ** 2, -1) / x.shape[-1], shape_ex)
        y = ops.broadcast_to(self.weight, x.shape) * (x_minus_E_x / ops.broadcast_to((Var_x + self.eps) ** 0.5, x.shape))
        y = y + ops.broadcast_to(self.bias, x.shape)
        return y
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            # MLIR tracing: pause trace during random mask generation
            # so the mask isn't recorded as a constant in the graph
            import needle.autograd as ag
            was_tracing = ag._TRACING
            if was_tracing and ag._ACTIVE_TRACER is not None:
                ag._ACTIVE_TRACER.pause()

            y = x * init.randb(*x.shape, p = 1 - self.p) / (1.0 - self.p)

            if was_tracing and ag._ACTIVE_TRACER is not None:
                ag._ACTIVE_TRACER.resume()
                # Mark the output as fallback since it contains random data
                ag._ACTIVE_TRACER.mark_fallback(y)
        else:
            y = x

        return y
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = self.fn(x) + x
        return y
        ### END YOUR SOLUTION


def _build_state_dict(module: Module, prefix: str = "") -> dict:
    """Recursively build a state_dict from a Module.

    Traverses __dict__ to find all Parameters, assigning hierarchical names
    like "conv1.weight", "bn1.bias".
    """
    import numpy as np
    result = {}
    for attr_name, value in module.__dict__.items():
        if attr_name.startswith('_'):
            continue
        param_prefix = f"{prefix}.{attr_name}" if prefix else attr_name
        if isinstance(value, Parameter):
            np_data = value.realize_cached_data()
            if hasattr(np_data, 'numpy'):
                result[param_prefix] = np_data.numpy()
            else:
                result[param_prefix] = np.array(np_data)
        elif isinstance(value, Module):
            result.update(_build_state_dict(value, param_prefix))
        elif isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                item_prefix = f"{param_prefix}.{idx}"
                if isinstance(item, Parameter):
                    np_data = item.realize_cached_data()
                    if hasattr(np_data, 'numpy'):
                        result[item_prefix] = np_data.numpy()
                    else:
                        result[item_prefix] = np.array(np_data)
                elif isinstance(item, Module):
                    result.update(_build_state_dict(item, item_prefix))
    return result


def _restore_state_dict(module: Module, state_dict: dict, prefix: str = "") -> None:
    """Recursively restore parameters from a state_dict into a Module.

    Raises ValueError if a parameter shape doesn't match.
    """
    import numpy as np
    for attr_name, value in module.__dict__.items():
        if attr_name.startswith('_'):
            continue
        param_prefix = f"{prefix}.{attr_name}" if prefix else attr_name
        if isinstance(value, Parameter):
            if param_prefix in state_dict:
                loaded_val = state_dict[param_prefix]
                if not isinstance(loaded_val, np.ndarray):
                    loaded_val = np.array(loaded_val, dtype=np.float32)
                # Shape check
                expected_shape = value.realize_cached_data().shape
                if loaded_val.shape != expected_shape:
                    raise ValueError(
                        f"Shape mismatch for '{param_prefix}': "
                        f"expected {expected_shape}, got {loaded_val.shape}"
                    )
                value.data = Tensor(loaded_val)
        elif isinstance(value, Module):
            _restore_state_dict(value, state_dict, param_prefix)
        elif isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                item_prefix = f"{param_prefix}.{idx}"
                if isinstance(item, Parameter):
                    if item_prefix in state_dict:
                        loaded_val = state_dict[item_prefix]
                        if not isinstance(loaded_val, np.ndarray):
                            loaded_val = np.array(loaded_val, dtype=np.float32)
                        item.data = Tensor(loaded_val)
                elif isinstance(item, Module):
                    _restore_state_dict(item, state_dict, item_prefix)
