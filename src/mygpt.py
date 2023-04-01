# curl https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt > tinyshakespeare.txt

with open("tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("corpus length:", len(text))

# print(text[:1000])

chars = sorted(list(set(text)))
vocab_size = len(chars)
print("".join(chars))
print(vocab_size)
