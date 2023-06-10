import torch


class LayerNorm1d:
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, x):
        # forward pass
        xmean = x.mean(dim=1, keepdim=True)  # batch mean
        xvar = x.var(dim=1, keepdim=True)  # batch variance
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)  # normalize
        self.out = self.gamma * xhat + self.beta  # scale and shift
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]
