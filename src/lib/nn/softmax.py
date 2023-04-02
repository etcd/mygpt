from torch import Tensor


def get_softmax(logits: Tensor):
    pows = logits.exp()
    return pows / pows.sum(1, keepdim=True)
