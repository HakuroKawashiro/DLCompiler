"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad * self.scalar


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        tmp_node = out_grad * node
        in_a = tmp_node * b / a
        in_b = tmp_node * log(a)
        return in_a, in_b
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * self.scalar * node / node.inputs[0]
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        in_a = out_grad / b
        in_b = -in_a * node
        return in_a, in_b
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # Use duck typing: Needle NDArray has permute(), numpy doesn't
        if hasattr(a, 'permute'):
            axis = [i for i in range(a.ndim)]

            if self.axes is None:
                axis[-1], axis[-2] = axis[-2], axis[-1]
            else:
                x, y = self.axes[0], self.axes[1]
                axis[x], axis[y] = axis[y], axis[x]

            return a.permute(tuple(axis))
        else:
            if self.axes is None:
                return array_api.swapaxes(a, -1, -2)

            return array_api.swapaxes(a, self.axes[0], self.axes[1])
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # Use duck typing: Needle NDArray has compact(), numpy doesn't
        if hasattr(a, 'compact'):
            return array_api.reshape(a.compact(), self.shape)
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        in_shape, out_shape = node.inputs[0].shape, node.shape
        in_shape_ex = tuple([1] * (len(out_shape) - len(in_shape)) + list(in_shape))
        axes = []
        for i in range(len(out_shape)):
            if in_shape_ex[i] != out_shape[i]:
                axes.append(i)
                
        ans = reshape(summation(out_grad, tuple(axes)) , in_shape)
        return ans
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis = self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        in_shape, out_shape = node.inputs[0].shape, node.shape
        in_shape_ex = []
        id = 0
        if self.axes is not None:
            if isinstance(self.axes, tuple) == False:
                self.axes = (self.axes,)
            for i in range(len(in_shape)):
                if i in self.axes or i - len(in_shape) in self.axes:
                    in_shape_ex.append(1)
                else:
                    in_shape_ex.append(out_shape[id])
                    id += 1
        else:
            in_shape_ex = [1]
        in_shape_ex = tuple(in_shape_ex)
        
        return broadcast_to(reshape(out_grad, in_shape_ex), in_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        ma, mb = out_grad @ transpose(b), transpose(a) @ out_grad
        
        if ma.shape != a.shape:
            a_shape_ex = tuple([1] * (len(ma.shape) - len(a.shape)) + list(a.shape))
            axes = []
            for i in range(len(a_shape_ex) - 2):
                if a_shape_ex[i] != ma.shape[i]:
                    axes.append(i)
            in_a = reshape(summation(ma, tuple(axes)), a.shape)
        else:
            in_a = ma
            
        if mb.shape != b.shape:
            b_shape_ex = tuple([1] * (len(mb.shape) - len(b.shape)) + list(b.shape))
            axes = []
            for i in range(len(b_shape_ex) - 2):
                if b_shape_ex[i] != mb.shape[i]:
                    axes.append(i)
            in_b = reshape(summation(mb, tuple(axes)), b.shape)
        else:
            in_b = mb
        
        return in_a, in_b
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad / a
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * node
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_data = node.inputs[0].realize_cached_data()
        mask = (input_data > 0)
        return out_grad * Tensor(mask, device=node.inputs[0].device)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (1 - node ** 2)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        shape = args[0].shape
        for arg in args:
            if arg.shape != shape:
                raise ValueError('The shape is invalid!')
        
        new_shape = list(shape)
        new_shape.insert(self.axis, len(args))
        
        res = array_api.empty(new_shape, device=args[0].device)
        for i, arg in enumerate(args):
            slices = [slice(None)] * len(new_shape)
            slices[self.axis] = i
            res[tuple(slices)] = arg
            
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        res = []
        new_shape = list(A.shape)
        new_shape.pop(self.axis)
        for i in range(A.shape[self.axis]):
            slices = [slice(None)] * len(A.shape)
            slices[self.axis] = i
            slice_result = A[tuple(slices)]
            if hasattr(slice_result, 'compact'):
                res.append(slice_result.compact().reshape(new_shape))
            else:
                res.append(numpy.ascontiguousarray(slice_result).reshape(new_shape))

        return tuple(res)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        tmp_axes = self.axes
        if isinstance(tmp_axes, int):
            tmp_axes = (tmp_axes,)
        if isinstance(tmp_axes, List):
            tmp_axes = tuple(tmp_axes)
        return array_api.flip(a, tmp_axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = tuple(a.shape[i] if i not in self.axes else a.shape[i] * (self.dilation + 1) for i in range(a.ndim))
        res = array_api.full(new_shape, 0, device=a.device)
        new_slices = [slice(None) if i not in self.axes else slice(None, None, self.dilation + 1) for i in range(a.ndim)]
        res[tuple(new_slices)] = a
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = tuple(a.shape[i] if i not in self.axes else a.shape[i] // (self.dilation + 1) for i in range(a.ndim))
        res = array_api.full(new_shape, 0, device=a.device)
        new_slices = [slice(None) if i not in self.axes else slice(None, None, self.dilation + 1) for i in range(a.ndim)]
        res = a[tuple(new_slices)]
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        # Needle NDArray path (uses strided views for efficiency)
        if hasattr(A, 'pad'):
            A_pad = A.pad(axes=((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
            N, H, W, C_in = A_pad.shape
            K, _, _, C_out = B.shape
            Ns, Hs, Ws, Cs = A_pad.strides
            X_shape = (N, (H - K) // self.stride + 1, (W - K) // self.stride + 1, K, K, C_in)
            X_strides = (Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs)
            X_2d_shape = (N * ((H - K) // self.stride + 1) * ((W - K) // self.stride + 1), K * K * C_in)
            X = A_pad.as_strided(shape=X_shape, strides=X_strides).compact().reshape(new_shape=X_2d_shape)
            Y = B.compact().reshape(new_shape=(K * K * C_in, C_out))
            Z = X @ Y
            return Z.compact().reshape(new_shape=(N, (H - K) // self.stride + 1, (W - K) // self.stride + 1, C_out))

        # NumPy fallback: im2col-based convolution
        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape
        pad = self.padding
        stride = self.stride

        A_pad = numpy.pad(A, ((0,0),(pad,pad),(pad,pad),(0,0)))
        H_out = (H + 2*pad - K) // stride + 1
        W_out = (W + 2*pad - K) // stride + 1

        # Reshape filter
        W_mat = B.reshape(K * K * C_in, C_out)

        # im2col + matmul
        result = numpy.zeros((N, H_out, W_out, C_out), dtype=A.dtype)
        for hi in range(H_out):
            for wi in range(W_out):
                patch = A_pad[:, hi*stride:hi*stride+K, wi*stride:wi*stride+K, :]
                patch_2d = patch.reshape(N, K * K * C_in)
                result[:, hi, wi, :] = patch_2d @ W_mat

        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, W = node.inputs
        K, _, _, _ = W.shape
        W_flip = flip(W, axes=(0, 1))
        W_flip_permuted = transpose(W_flip, axes=(2, 3))
        out_grad_dilated = dilate(out_grad, axes=(1, 2), dilation=self.stride - 1)
        dX = conv(out_grad_dilated, W_flip_permuted, stride=1, padding=K - 1 - self.padding)
        
        out_grad_dilated_permuted = transpose(out_grad_dilated, axes=(0, 2))
        out_grad_dilated_permuted = transpose(out_grad_dilated_permuted, axes=(0, 1))
        X_permuted = transpose(X, axes=(0, 3))
        dW = conv(X_permuted, out_grad_dilated_permuted, stride=1, padding=self.padding)
        dW = transpose(dW, axes=(0, 2))
        dW = transpose(dW, axes=(0, 1))
        return dX, dW
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


