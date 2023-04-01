
from basic_tokenizer import make_encoder, make_decoder

with open("sample_data/tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("corpus size", len(text))

alphabet = sorted(list(set(text)))
alphabet_size = len(alphabet)

print("Alphabet", "".join(alphabet))
print("Alphabet size", alphabet_size)

encode = make_encoder(alphabet)
decode = make_decoder(alphabet)
