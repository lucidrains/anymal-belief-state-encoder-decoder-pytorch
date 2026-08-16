import torch
from torch import nn

from ema_pytorch import EMATensor

# running statistics for normalizing observations fed to the teacher
# maintained as exponential moving averages, courtesy of ema-pytorch

class RunningStats(nn.Module):
    def __init__(
        self,
        shape,
        beta = 0.99,
        eps = 1e-5
    ):
        super().__init__()
        shape = shape if isinstance(shape, tuple) else (shape,)
        self.eps = eps

        ema_kwargs = dict(beta = beta, update_after_step = 0, update_every = 1, allow_different_devices = True)

        self.mean_ema = EMATensor(torch.zeros(shape), **ema_kwargs)
        self.mean_squared_ema = EMATensor(torch.zeros(shape), **ema_kwargs)

    def clear(self):
        for ema in (self.mean_ema, self.mean_squared_ema):
            ema.ema_pytree.zero_()
            ema.initted.data.copy_(torch.tensor(False))

    def push(self, x):
        x = x.detach()

        for ema, value in ((self.mean_ema, x), (self.mean_squared_ema, x ** 2)):
            ema.ema_pytree = ema.ema_pytree.to(x.device)
            ema.pytree.copy_(value)
            ema.update()

    def mean(self):
        return self.mean_ema()

    def variance(self):
        return (self.mean_squared_ema() - self.mean() ** 2).clamp(min = 0)

    def rstd(self):
        return torch.rsqrt(self.variance() + self.eps)

    def norm(self, x):
        return (x - self.mean()) * self.rstd()
