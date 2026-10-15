import re
import random
from collections import defaultdict, Counter


class BigramModel:
    def __init__(self, corpus, frequency_threshold=None):
        """corpus: a list of text strings used to train the bigram 
model."""
        text = " ".join(corpus)
        self.vocab, self.bigram_probs = self._analyze_bigrams(text, 
frequency_threshold)

    def _simple_tokenizer(self, text, frequency_threshold=None):
        tokens = re.findall(r"\b\w+\b", text.lower())
        if not frequency_threshold:
            return tokens
        word_counts = Counter(tokens)
        return [t for t in tokens if word_counts[t] >= 
frequency_threshold]

    def _analyze_bigrams(self, text, frequency_threshold=None):
        words = self._simple_tokenizer(text, frequency_threshold)
        bigrams = list(zip(words[:-1], words[1:]))

        bigram_counts = Counter(bigrams)
        unigram_counts = Counter(words)

        bigram_probs = defaultdict(dict)
        for (word1, word2), count in bigram_counts.items():
            bigram_probs[word1][word2] = count / unigram_counts[word1]

        return list(unigram_counts.keys()), bigram_probs

    def generate_text(self, start_word, length=20):
        """Generate text based on bigram probabilities starting from 
start_word."""
        current_word = start_word.lower()
        generated_words = [current_word]

        for _ in range(length - 1):
            next_words = self.bigram_probs.get(current_word)
            if not next_words:
                break
            next_word = random.choices(
                list(next_words.keys()), weights=next_words.values()
            )[0]
            generated_words.append(next_word)
            current_word = next_word

        return " ".join(generated_words)
