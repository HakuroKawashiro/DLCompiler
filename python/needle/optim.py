"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None

    def clip_grad_norm(self, max_norm: float = 0.25) -> float:
        """Clip the gradient norm of all parameters.

        Computes the total L2 norm of all parameter gradients.
        If total_norm > max_norm, scales all gradients by max_norm / total_norm.

        Args:
            max_norm: Maximum allowed gradient norm.

        Returns:
            The total norm before clipping (0.0 if no gradients).
        """
        total_norm_sq = 0.0
        for p in self.params:
            if p.grad is not None:
                grad_data = p.grad.realize_cached_data()
                total_norm_sq += float((grad_data ** 2).sum())

        total_norm = total_norm_sq ** 0.5

        if total_norm > max_norm and total_norm > 0:
            scale = max_norm / total_norm
            for p in self.params:
                if p.grad is not None:
                    p.grad = p.grad * scale  # Scale in-place via Tensor.__mul__

        return total_norm


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            pny_theta = param.grad.data + self.weight_decay * param.data
            
            if param not in self.u:
                self.u[param] = (1.0 - self.momentum) * pny_theta.data
            else:
                self.u[param].data = self.momentum * self.u[param].data + (1.0 - self.momentum) * pny_theta.data
            
            param.data = param.data + (-self.lr) * self.u[param].data 
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        return Optimizer.clip_grad_norm(self, max_norm)
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param in self.params:
            pny_theta = param.grad.data + self.weight_decay * param.data
            
            if param not in self.m:
                self.m[param] = (1.0 - self.beta1) * pny_theta.data
            else:
                self.m[param].data = self.beta1 * self.m[param].data + (1 - self.beta1) * pny_theta.data
                
            m_hat = self.m[param].data / (1.0 - self.beta1 ** self.t)
            
            if param not in self.v:
                self.v[param] = (1.0 - self.beta2) * (pny_theta.data ** 2).data
            else:
                self.v[param].data = self.beta2 * self.v[param].data + (1 - self.beta2) * (pny_theta.data ** 2).data
               
            v_hat = self.v[param].data / (1.0 - self.beta2 ** self.t)
            
            param.data = param.data + (-self.lr) * (m_hat.data / ((v_hat.data ** 0.5).data + self.eps).data).data
        ### END YOUR SOLUTION
