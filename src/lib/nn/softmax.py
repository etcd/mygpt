from torch import Tensor


def get_softmax(logits: Tensor, dim=1):
    pows = logits.exp()
    return pows / pows.sum(dim, keepdim=True)
