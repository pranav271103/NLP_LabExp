# Import required libraries
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, TweetTokenizer, RegexpTokenizer

# Ensure necessary resources are downloaded
nltk.download('punkt')

# Function to read file and tokenize text
def tokenize_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Sentence tokenization
    sentences = sent_tokenize(text)

    # Word tokenization (default)
    words_default = word_tokenize(text)

    # Tweet tokenizer for social media text
    tweet_tokenizer = TweetTokenizer().tokenize(text)

    # Regular expression tokenizer (splits by word characters)
    regex_tokenizer = RegexpTokenizer(r'\w+').tokenize(text)

    # Create formatted output
    print(f"{'Tokenizer':<25} | Example Output")
    print("-" * 70)
    print(f"{'Sentence Tokenizer':<25} | {sentences[:3]}")
    print(f"{'Word Tokenizer (default)':<25} | {words_default[:10]}")
    print(f"{'Tweet Tokenizer':<25} | {tweet_tokenizer[:10]}")
    print(f"{'Regexp Tokenizer':<25} | {regex_tokenizer[:10]}")

# Example text file creation for demonstration
sample_text = """Hello world! This is an example text for testing tokenization.
Email me at example@test.com. Let's explore NLTK tokenizers for sentence and word-level tokenization."""
with open("sample_text.txt", "w", encoding="utf-8") as f:
    f.write(sample_text)

# Execute the tokenizer
tokenize_text("sample_text.txt")
