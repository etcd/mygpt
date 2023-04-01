# curl https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt > tinyshakespeare.txt

from basic_tokenizer import make_encoder, make_decoder

with open("src/tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("corpus length:", len(text))

# print(text[:1000])

alphabet = sorted(list(set(text)))
alphabet_size = len(alphabet)
print("".join(alphabet))
print(alphabet_size)

encode = make_encoder(alphabet)
decode = make_decoder(alphabet)

print(encode("hello"))
print(decode(encode("hello")))
