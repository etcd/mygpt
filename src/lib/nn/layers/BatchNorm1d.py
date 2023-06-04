import torch


class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # params trained by gradient descent
        self.gamma = torch.ones(dim)  # scale
        self.beta = torch.zeros(dim)  # shift
        # buffers trained by moving average
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        # forward pass
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0, 1)
            x_mean = x.mean(dim, keepdim=True)  # batch mean
            x_var = x.var(dim, keepdim=True)  # batch var
        else:
            x_mean = self.running_mean
            x_var = self.running_var

        # normalize to unit variance
        x_hat = (x - x_mean) / torch.sqrt(x_var + self.eps)
        self.out = self.gamma * x_hat + self.beta  # scale and shift

        # update running mean and var
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * \
                    self.running_mean + self.momentum * x_mean
                self.running_var = (1 - self.momentum) * \
                    self.running_var + self.momentum * x_var
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]
