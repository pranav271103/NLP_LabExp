import nltk
from nltk.corpus import brown
from collections import defaultdict, Counter
import math

def create_bigram_model():
    """Create a basic bigram model from the Brown corpus."""
    # Download the Brown corpus if not already downloaded
    try:
        nltk.data.find('corpora/brown')
    except LookupError:
        nltk.download('brown')
    
    # Get sentences and add start/end markers
    sentences = brown.sents()
    words = [word.lower() for sent in sentences for word in sent]
    
    # Create unigram and bigram counts
    unigram_counts = Counter(words)
    bigram_counts = defaultdict(int)
    
    # Count bigrams
    for i in range(len(words) - 1):
        bigram = (words[i], words[i+1])
        bigram_counts[bigram] += 1
    
    return unigram_counts, bigram_counts, len(words)

def add_one_smoothing(unigram_counts, bigram_counts, vocab_size):
    """Apply add-one smoothing to the bigram model."""
    smoothed_probs = {}
    
    # Convert unigram counts to a list of words
    vocabulary = list(unigram_counts.keys())
    
    # Calculate smoothed probabilities
    for w1 in vocabulary:
        for w2 in vocabulary:
            bigram = (w1, w2)
            # Add-one smoothing formula: (C(w1,w2) + 1) / (C(w1) + V)
            # where V is the vocabulary size
            numerator = bigram_counts.get(bigram, 0) + 1
            denominator = unigram_counts[w1] + vocab_size
            smoothed_prob = numerator / denominator
            smoothed_probs[bigram] = smoothed_prob
    
    return smoothed_probs

def main():
    print("Creating bigram model from Brown corpus...")
    unigram_counts, bigram_counts, total_words = create_bigram_model()
    vocab_size = len(unigram_counts)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of unique bigrams: {len(bigram_counts)}")
    
    # Get some example bigrams before smoothing
    example_bigrams = list(bigram_counts.items())[:5]
    print("\nExample bigrams (before smoothing):")
    for (w1, w2), count in example_bigrams:
        print(f"'{w1} {w2}': {count}")
    
    # Apply add-one smoothing
    print("\nApplying add-one smoothing...")
    smoothed_probs = add_one_smoothing(unigram_counts, bigram_counts, vocab_size)
    
    # Display some example smoothed probabilities
    print("\nExample smoothed probabilities (log scale):")
    for (w1, w2), count in example_bigrams:
        prob = smoothed_probs.get((w1, w2), 0)
        print(f"P('{w2}'|'{w1}') = {prob:.6f} (log: {math.log(prob):.2f})")
    
    # Find and show some zero-probability bigrams that now have non-zero probability
    print("\nPreviously unseen bigrams now have non-zero probability:")
    unseen_examples = 0
    for (w1, w2), prob in smoothed_probs.items():
        if (w1, w2) not in bigram_counts and prob > 0:
            print(f"P('{w2}'|'{w1}') = {prob:.6f}")
            unseen_examples += 1
            if unseen_examples >= 3:  # Show just 3 examples
                break

if __name__ == "__main__":
    main()
