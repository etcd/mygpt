from torch import Tensor


def get_softmax(logits: Tensor):
    counts = logits.exp()
    return counts / counts.sum(1, keepdim=True)
