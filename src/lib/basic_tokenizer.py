# more sophisticated tokenizers are available online, e.g.:
# SentencePiece by Google
# TikToken by OpenAI


def make_encoder(alphabet, other=[]):
    def encode(str):
        other_chars = {ch: len(alphabet)+i for i, ch in enumerate(other)}
        stoi = {ch: i for i, ch in enumerate(alphabet)} | other_chars
        return [stoi[ch] for ch in str]
    return encode


def make_decoder(alphabet, other=[]):
    def decode(ints):
        other_chars = {len(alphabet)+i: ch for i, ch in enumerate(other)}
        itos = {i: ch for i, ch in enumerate(alphabet)} | other_chars
        return "".join([itos[i] for i in ints])
    return decode
