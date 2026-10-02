from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        if isinstance(Z, NDArray):
            z_mx = Z.max(axis = (1,), keepdims = True)
        else:
            z_mx = array_api.max(Z, axis = (1,), keepdims = True)
        
        res = array_api.log(array_api.sum(array_api.exp(Z - array_api.broadcast_to(z_mx, Z.shape)), axis = (1,), keepdims = True))
        res = array_api.broadcast_to(res + z_mx, Z.shape)
        return Z - res
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        in_shape_ex = list(node.inputs[0].shape)
        in_shape_ex[1] = 1
        in_shape_ex = tuple(in_shape_ex)
        exp_z = exp(node.inputs[0])
        return out_grad - broadcast_to(reshape(summation(out_grad, (1,)) / summation(exp_z, (1,)), in_shape_ex), node.shape) * exp_z
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        if isinstance(Z, NDArray):
            z_mx = Z.max(axis = self.axes, keepdims = True)
        else:
            z_mx = array_api.max(Z, axis = self.axes, keepdims = True)
        
        res = array_api.log(array_api.sum(array_api.exp(Z - array_api.broadcast_to(z_mx, Z.shape)), axis = self.axes))
        zmx_new_shape = []
        old_axes = self.axes
        if isinstance(old_axes, tuple) is False:
            old_axes = (old_axes,)
        for i, val in enumerate(z_mx.shape):
            if i not in old_axes:
                zmx_new_shape.append(val)
        res = res + array_api.reshape(z_mx, tuple(zmx_new_shape))
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        in_shape, out_shape = node.inputs[0].shape, node.shape
        in_shape_ex = []
        id = 0
        if self.axes is not None:
            if isinstance(self.axes, tuple) == False:
                self.axes = (self.axes,)
            for i in range(len(in_shape)):
                if i in self.axes:
                    in_shape_ex.append(1)
                else:
                    in_shape_ex.append(out_shape[id])
                    id += 1
        in_shape_ex = tuple(in_shape_ex)
        
        og_ex = broadcast_to(reshape(out_grad, in_shape_ex), in_shape)
        Z_m_mx_exp = exp(node.inputs[0] - broadcast_to(reshape(node, in_shape_ex), in_shape))
        return og_ex * Z_m_mx_exp
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)