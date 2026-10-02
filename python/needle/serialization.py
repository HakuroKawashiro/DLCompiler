"""Model serialization — save/load model parameters."""

import pickle
import logging
from typing import Any, Dict

from .autograd import Tensor
from .nn.nn_basic import Module, Parameter

logger = logging.getLogger("needle.serialization")


def save(model: Module, path: str) -> None:
    """Save model parameters to a file.

    Args:
        model: A needle nn.Module instance
        path: File path to save to (pickle format)
    """
    state = model.state_dict() if hasattr(model, 'state_dict') else _state_dict(model)
    with open(path, 'wb') as f:
        pickle.dump(state, f)
    logger.info(f"Model saved to {path}")


def load(model: Module, path: str) -> Module:
    """Load model parameters from a file.

    Args:
        model: A needle nn.Module instance (architecture must match saved model)
        path: File path to load from

    Returns:
        The model with loaded parameters (same object, mutated in place)
    """
    with open(path, 'rb') as f:
        state = pickle.load(f)

    if hasattr(model, 'load_state_dict'):
        model.load_state_dict(state)
    else:
        _load_state_dict(model, state)

    logger.info(f"Model loaded from {path}")
    return model


def _state_dict(module: Module, prefix: str = "") -> Dict[str, Any]:
    """Recursively build a state_dict {param_name: numpy_array} from a Module.

    Uses the same traversal logic as _name_modules in compiler.py.
    """
    result: Dict[str, Any] = {}
    import numpy as np

    for attr_name, value in module.__dict__.items():
        if attr_name.startswith('_'):
            continue

        param_prefix = f"{prefix}.{attr_name}" if prefix else attr_name

        if isinstance(value, Parameter):
            np_data = value.realize_cached_data()
            # Convert to plain numpy array for serialization
            if hasattr(np_data, 'numpy'):
                result[param_prefix] = np_data.numpy()
            else:
                result[param_prefix] = np.array(np_data)
        elif isinstance(value, Module):
            result.update(_state_dict(value, param_prefix))
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
                    result.update(_state_dict(item, item_prefix))

    return result


def _load_state_dict(module: Module, state_dict: Dict[str, Any], prefix: str = "") -> None:
    """Recursively restore parameters from a state_dict into a Module."""
    import numpy as np

    for attr_name, value in module.__dict__.items():
        if attr_name.startswith('_'):
            continue

        param_prefix = f"{prefix}.{attr_name}" if prefix else attr_name

        if isinstance(value, Parameter):
            if param_prefix in state_dict:
                loaded_val = state_dict[param_prefix]
                if isinstance(loaded_val, np.ndarray):
                    value.data = Tensor(loaded_val)
                else:
                    value.data = Tensor(np.array(loaded_val, dtype=np.float32))
            else:
                logger.warning(f"Parameter '{param_prefix}' not found in state_dict")
        elif isinstance(value, Module):
            _load_state_dict(value, state_dict, param_prefix)
        elif isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                item_prefix = f"{param_prefix}.{idx}"
                if isinstance(item, Parameter):
                    if item_prefix in state_dict:
                        loaded_val = state_dict[item_prefix]
                        if isinstance(loaded_val, np.ndarray):
                            item.data = Tensor(loaded_val)
                        else:
                            item.data = Tensor(np.array(loaded_val, dtype=np.float32))
                elif isinstance(item, Module):
                    _load_state_dict(item, state_dict, item_prefix)
