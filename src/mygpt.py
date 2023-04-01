
from basic_tokenizer import make_encoder, make_decoder

with open("sample_data/tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("corpus length:", len(text))

alphabet = sorted(list(set(text)))
alphabet_size = len(alphabet)
print("".join(alphabet))
print(alphabet_size)

encode = make_encoder(alphabet)
decode = make_decoder(alphabet)

print(encode("hello"))
print(decode(encode("hello")))
