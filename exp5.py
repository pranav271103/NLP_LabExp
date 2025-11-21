from collections import defaultdict

def add_one_smoothing(bigram_counts, unigrams):
    """
    Perform add-one (Laplace) smoothing on sparse bigram counts.
    
    Args:
        bigram_counts (dict): Nested dictionary of bigram counts {w1: {w2: count}}.
        unigrams (dict): Dictionary of unigram counts {w: count}.

    Returns:
        dict: Smoothed bigram probabilities.
    """
    vocab = set(unigrams.keys())
    V = len(vocab)
    smoothed_probs = defaultdict(dict)
    
    for w1 in vocab:
        total_count = unigrams[w1] + V  # denominator includes +V for smoothing
        for w2 in vocab:
            count = bigram_counts[w1].get(w2, 0)
            smoothed_probs[w1][w2] = (count + 1) / total_count
    
    return smoothed_probs

# Example usage
if __name__ == "__main__":
    unigrams = {'I': 3, 'love': 2, 'Python': 1}
    bigram_counts = {
        'I': {'love': 2, 'Python': 0},
        'love': {'Python': 1, 'I': 0},
        'Python': {'I': 0, 'love': 0}
    }

    smoothed = add_one_smoothing(bigram_counts, unigrams)

    for w1 in smoothed:
        for w2 in smoothed[w1]:
            print(f"P({w2}|{w1}) = {smoothed[w1][w2]:.4f}")
