import nltk
from nltk.corpus import brown
from nltk import bigrams, trigrams
from collections import Counter

def analyze_ngrams():
    # Download the Brown corpus if not already downloaded
    try:
        nltk.data.find('corpora/brown')
    except LookupError:
        nltk.download('brown')
    
    # Get all words from the Brown corpus
    words = brown.words()
    
    # Generate bigrams and trigrams
    print("Generating n-grams...")
    bigram_list = list(bigrams(words))
    trigram_list = list(trigrams(words))
    
    # Count frequencies
    bigram_freq = Counter(bigram_list)
    trigram_freq = Counter(trigram_list)
    
    # Sort by frequency (most common first)
    sorted_bigrams = bigram_freq.most_common()
    sorted_trigrams = trigram_freq.most_common()
    
    # Function to save n-grams to file
    def save_ngrams(ngrams, filename, n):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"{n}-gram,Count\n")
            for ngram, count in ngrams:
                f.write(f"{' '.join(ngram)},{count}\n")
    
    # Save results to files
    save_ngrams(sorted_bigrams, 'bigrams_frequency.csv', 2)
    save_ngrams(sorted_trigrams, 'trigrams_frequency.csv', 3)
    
    # Print summary
    print(f"\nAnalysis Complete!")
    print(f"Total unique bigrams found: {len(bigram_freq)}")
    print(f"Total unique trigrams found: {len(trigram_freq)}")
    print("\nTop 10 most frequent bigrams:")
    for i, (bigram, count) in enumerate(sorted_bigrams[:10], 1):
        print(f"{i}. {' '.join(bigram)}: {count}")
    
    print("\nTop 10 most frequent trigrams:")
    for i, (trigram, count) in enumerate(sorted_trigrams[:10], 1):
        print(f"{i}. {' '.join(trigram)}: {count}")
    
    print("\nResults have been saved to 'bigrams_frequency.csv' and 'trigrams_frequency.csv'")

if __name__ == "__main__":
    analyze_ngrams()
