import math
from lib.basic_tokenizer import make_tokenizers
from lib.bigrams.calculate_likelihood import calculate_likelihood
from lib.bigrams.count import count_bigrams
from lib.bigrams.generate_word import generate_word
from lib.math.get_perplexity import get_perplexity
# from lib.bigrams.plot import plot_bigrams

words = open('sample_data/names.txt', 'r').read().splitlines()
alphabet = sorted(list(set(''.join(words))))
(encode, decode) = make_tokenizers(alphabet, ['.'])

bigrams = count_bigrams(words, encode)
# plot_bigrams(bigrams, decode)

bigrams_plus_one = bigrams + 1  # model smoothing with +1
probabilities = bigrams_plus_one/bigrams_plus_one.sum(1, keepdim=True)

# for _ in range(5):
#     print(generate_word(probabilities, decode))

(log_likelihood, n) = calculate_likelihood(probabilities, words, encode)

print(f"Average NLL: {-log_likelihood/n:.4f}")
print(f"Perplexity: {get_perplexity(log_likelihood, n):.4f}")
