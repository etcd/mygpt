
import torch

from lib.basic_tokenizer import make_tokenizers


def split_list(list, fraction):
    n = int(len(list) * fraction)
    return [list[:n], list[n:]]


def get_batch(data, block_size, batch_size):
    '''Gets a random batch of blocks from data.
    '''
    idx = torch.randint(len(data)-block_size, (batch_size,))
    blocks = torch.stack([data[i:i+block_size] for i in idx])
    targetss = torch.stack([data[i+1:i+block_size+1] for i in idx])
    return blocks, targetss


with open("sample_data/tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()
alphabet = sorted(list(set(text)))
[encode, decode] = make_tokenizers(alphabet)

data = torch.tensor(encode(text), dtype=torch.long)
[train_data, validate_data] = split_list(data, 0.9)

block_size = 8
batch_size = 4

blocks, targetss = get_batch(train_data, block_size, batch_size)

token_embedding_table = torch.nn.Embedding(
    len(alphabet), len(alphabet))  # batch, time, channels (vocab size)


def forward(token_embedding_table, blocks, targetss=None):
    logits = token_embedding_table(blocks)
    if targetss is None:
        loss = None
    else:
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, len(alphabet)), targetss.view(-1))
    return logits, loss


def generate(idx, targetss, max_new_tokens):
    # idx is (B, T) array of indices in current block
    for _ in range(max_new_tokens):
        # get predictions
        logits, loss = forward(token_embedding_table, idx, targetss)
        # get last token
        logits = logits[:, -1, :]  # (B, C)
        # apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=1)  # (B, C)
        # sample from probabilities
        idx_next = torch.multinomial(probabilities, num_samples=1)  # (B, 1)
        # append to idx
        idx = torch.cat([idx, idx_next], dim=1)  # (B, T+1)
    return idx


logits, loss = forward(token_embedding_table, blocks, targetss)
print(logits)

# idx = torch.zeros((1, 1), dtype=torch.long)
# print(decode(generate(idx, targetss=None,
#       max_new_tokens=100)[0].tolist()))
