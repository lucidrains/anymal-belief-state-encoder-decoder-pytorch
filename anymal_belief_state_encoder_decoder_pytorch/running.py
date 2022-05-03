import torch
from torch import nn

class RunningStats(nn.Module):
    def __init__(self, shape, eps = 1e-5):
        super().__init__()
        shape = shape if isinstance(shape, tuple) else (shape,)

        self.shape = shape
        self.eps = eps
        self.n = 0

        self.register_buffer('old_mean', torch.zeros(shape), persistent = False)
        self.register_buffer('new_mean', torch.zeros(shape), persistent = False)
        self.register_buffer('old_std', torch.zeros(shape), persistent = False)
        self.register_buffer('new_std', torch.zeros(shape), persistent = False)

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_mean.copy_(x.data)
            self.new_mean.copy_(x.data)
            self.old_std.zero_()
            self.new_std.zero_()
            return

        self.new_mean.copy_(self.old_mean + (x - self.old_mean) / self.n)
        self.new_std.copy_(self.old_std + (x - self.old_mean) * (x - self.new_mean))

        self.old_mean.copy_(self.new_mean)
        self.old_std.copy_(self.new_std)

    def mean(self):
        return self.new_mean if self.n else torch.zeros_like(self.new_mean)

    def variance(self):
        return (self.new_std / (self.n - 1)) if self.n > 1 else torch.zeros_like(self.new_std)

    def rstd(self):
        return torch.rsqrt(self.variance() + self.eps)

    def norm(self, x):
        return (x - self.mean()) * self.rstd()
