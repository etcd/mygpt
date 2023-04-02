from lib.basic_tokenizer import make_tokenizers
from lib.bigrams.get_log_likelihood import get_log_likelihood
from lib.bigrams.count import count_bigrams
from lib.math.get_perplexity import get_perplexity

words = open('sample_data/names.txt', 'r').read().splitlines()
alphabet = ['.'] + sorted(list(set(''.join(words))))
alphabet_size = len(alphabet)
(encode, decode) = make_tokenizers(alphabet)

encoded_words = [encode('.' + word + '.') for word in words]
bigrams = count_bigrams(encoded_words, alphabet_size)
# plot_bigrams(bigrams, decode, alphabet_size)

bigrams_plus_one = bigrams + 1  # model smoothing with +1
probabilities = bigrams_plus_one/bigrams_plus_one.sum(1, keepdim=True)

# for _ in range(5):
#     print(generate_word(probabilities, decode))

log_likelihood = 0.0
num_bigrams = 0
for word in words:
    encoded_word = encode('.' + word + '.')
    word_ll = get_log_likelihood(probabilities, encoded_word)
    log_likelihood += word_ll
    num_bigrams += len(word)+1

print(f"Average NLL: {-log_likelihood/num_bigrams:.4f}")
print(f"Perplexity: {get_perplexity(log_likelihood, num_bigrams):.4f}")
