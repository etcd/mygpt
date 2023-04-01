
from basic_tokenizer import make_encoder, make_decoder
import torch

with open("sample_data/tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("corpus size", len(text))

alphabet = sorted(list(set(text)))
alphabet_size = len(alphabet)

print("Alphabet", "".join(alphabet))
print("Alphabet size", alphabet_size)

encode = make_encoder(alphabet)
decode = make_decoder(alphabet)

data = torch.tensor(encode(text), dtype=torch.long)

print(data.shape, data.dtype)
print(data[:1000])
