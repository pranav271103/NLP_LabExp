import nltk
from nltk.corpus import words
from nltk.tokenize import word_tokenize

# Download required resources
nltk.download('words')
nltk.download('punkt')

def find_unusual_words(text):
    english_vocab = set(w.lower() for w in words.words())
    text_vocab = set(w.lower() for w in word_tokenize(text) if w.isalpha())
    unusual = text_vocab.difference(english_vocab)
    return sorted(unusual)

# Example usage
sample_text = "The quick brown fox jumps over the lazy doge and quizzaciously zapped the xylophonist."
print(find_unusual_words(sample_text))
