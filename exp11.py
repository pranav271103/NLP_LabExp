import argparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import jaccard as jaccard_distance
import spacy
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    """Preprocess text by removing punctuation and stopwords, and converting to lowercase"""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def cosine_sim(text1, text2):
    """Calculate cosine similarity between two texts using TF-IDF"""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

def jaccard_sim(text1, text2):
    """Calculate Jaccard similarity between two texts"""
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def levenshtein_distance(text1, text2):
    """Calculate Levenshtein distance between two texts"""
    size_x = len(text1) + 1
    size_y = len(text2) + 1
    matrix = np.zeros((size_x, size_y))
    
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y
    
    for x in range(1, size_x):
        for y in range(1, size_y):
            if text1[x-1] == text2[y-1]:
                matrix[x, y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1] + 1,
                    matrix[x, y-1] + 1
                )
    
    return matrix[size_x - 1, size_y - 1]

def ngram_similarity(text1, text2, n=2):
    """Calculate n-gram similarity between two texts"""
    ngrams1 = set(ngrams(text1.split(), n))
    ngrams2 = set(ngrams(text2.split(), n))
    
    intersection = len(ngrams1.intersection(ngrams2))
    union = len(ngrams1.union(ngrams2))
    
    return intersection / union if union != 0 else 0

def word_embedding_similarity(text1, text2):
    """Calculate similarity using word embeddings (spaCy)"""
    try:
        nlp = spacy.load('en_core_web_md')
    except OSError:
        print("Downloading spaCy model 'en_core_web_md'...")
        from spacy.cli import download
        download('en_core_web_md')
        nlp = spacy.load('en_core_web_md')
    
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    return doc1.similarity(doc2)

def compare_sentences(sentence1, sentence2):
    """Compare two sentences using multiple similarity metrics"""
    # Preprocess the sentences
    preprocessed1 = preprocess_text(sentence1)
    preprocessed2 = preprocess_text(sentence2)
    
    # Calculate similarities
    similarities = {
        'Cosine Similarity (TF-IDF)': cosine_sim(preprocessed1, preprocessed2),
        'Jaccard Similarity': jaccard_sim(preprocessed1, preprocessed2),
        'Levenshtein Distance': levenshtein_distance(sentence1, sentence2),
        'Bigram Similarity': ngram_similarity(preprocessed1, preprocessed2, n=2),
        'Word Embedding Similarity': word_embedding_similarity(sentence1, sentence2)
    }
    
    return similarities

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Compare two sentences using various text similarity metrics.')
    parser.add_argument('sentence1', type=str, help='First sentence to compare')
    parser.add_argument('sentence2', type=str, help='Second sentence to compare')
    args = parser.parse_args()
    
    # Get the sentences from command line arguments
    sentence1 = args.sentence1
    sentence2 = args.sentence2
    
    print(f"\nComparing sentences:")
    print(f"1: {sentence1}")
    print(f"2: {sentence2}")
    
    # Calculate similarities
    print("\nSimilarity Metrics:")
    print("-" * 60)
    
    similarities = compare_sentences(sentence1, sentence2)
    
    # Display results
    max_name_length = max(len(name) for name in similarities.keys())
    
    for metric, score in similarities.items():
        if 'Distance' in metric:
            # For distance metrics, lower is better
            print(f"{metric:{max_name_length}} : {score:.4f} (lower is more similar)")
        else:
            # For similarity metrics, higher is better
            print(f"{metric:{max_name_length}} : {score:.4f} (higher is more similar)")

if __name__ == "__main__":
    main()
