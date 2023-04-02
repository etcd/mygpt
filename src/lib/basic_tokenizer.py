# more sophisticated tokenizers are available online, e.g.:
# SentencePiece by Google
# TikToken by OpenAI


def make_tokenizers(alphabet: list[str], special: list[str] = []):
    len_special = len(special)

    def encode(s: str):
        other_chars = {ch: i for i, ch in enumerate(special)}
        stoi = other_chars | {
            ch: i+len_special for i, ch in enumerate(alphabet)}
        return [stoi[ch] for ch in s]

    def decode(ints: list[int]):
        other_chars = {i: ch for i, ch in enumerate(special)}
        itos = other_chars | {
            i+len_special: ch for i, ch in enumerate(alphabet)}
        return "".join([itos[i] for i in ints])

    return (encode, decode)
