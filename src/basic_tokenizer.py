# more sophisticated tokenizers are available online, e.g.:
# SentencePiece by Google
# TikToken by OpenAI

def make_serializers(text):
    alphabet = sorted(list(set(text)))
    return make_encoder(alphabet), make_decoder(alphabet)


def make_encoder(alphabet):
    def encode(str):
        stoi = {ch: i for i, ch in enumerate(alphabet)}
        return [stoi[ch] for ch in str]
    return encode


def make_decoder(alphabet):
    def decode(ints):
        itos = {i: ch for i, ch in enumerate(alphabet)}
        return "".join([itos[i] for i in ints])
    return decode
