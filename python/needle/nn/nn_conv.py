"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_channels, out_channels, shape=(kernel_size, kernel_size, in_channels, out_channels), device=device, dtype=dtype))
        bias_bound = 1.0 / np.sqrt(in_channels * kernel_size * kernel_size)
        self.bias = Parameter(init.rand(out_channels, low=-bias_bound, high=bias_bound, device=device, dtype=dtype)) if bias else None
        self.padding = (kernel_size - 1) // 2
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x = ops.transpose(ops.transpose(x, (1, 3)), (1, 2))  # NCHW -> NHWC
        conv_x = ops.conv(x, self.weight, stride=self.stride, padding=self.padding)
        if self.bias is not None:
            conv_x = conv_x + ops.broadcast_to(ops.reshape(self.bias, (1, 1, 1, self.out_channels)), conv_x.shape)
        out = ops.transpose(ops.transpose(conv_x, (1, 3)), (2, 3))  # NHWC -> NCHW
        return out
        ### END YOUR SOLUTION