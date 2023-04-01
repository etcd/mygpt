
from basic_tokenizer import make_encoder, make_decoder
import torch


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
encode = make_encoder(alphabet)
decode = make_decoder(alphabet)
data = torch.tensor(encode(text), dtype=torch.long)

[train_data, validate_data] = split_list(data, 0.9)

block_size = 8
batch_size = 4

blocks, targetss = get_batch(train_data, block_size, batch_size)
print('blocks', blocks)
print('targetss', targetss)

for i in range(batch_size):
    block = blocks[i]
    targets = targetss[i]
    print("in", block, "out", targets)
