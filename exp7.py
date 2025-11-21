import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)  # Fix for LookupError in newer NLTK versions
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

def process_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # 1. Tokenization
    words = word_tokenize(text)

    # 2. POS tagging
    pos_tags = nltk.pos_tag(words)

    # 3. Stop-word removal
    stop_words = set(stopwords.words('english'))
    filtered_words = [w for w in words if w.lower() not in stop_words and w.isalpha()]

    # 4. Stemming using different stemmers
    porter = PorterStemmer()
    lancaster = LancasterStemmer()
    snowball = SnowballStemmer('english')

    porter_stems = [porter.stem(w) for w in filtered_words]
    lancaster_stems = [lancaster.stem(w) for w in filtered_words]
    snowball_stems = [snowball.stem(w) for w in filtered_words]

    # 5. Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(w) for w in filtered_words]

    # Display results
    print(f"{'Process':<25} | Example Output")
    print("-" * 70)
    print(f"{'POS Tagging':<25} | {pos_tags[:10]}")
    print(f"{'Stop-word Removal':<25} | {filtered_words[:10]}")
    print(f"{'Porter Stemmer':<25} | {porter_stems[:10]}")
    print(f"{'Lancaster Stemmer':<25} | {lancaster_stems[:10]}")
    print(f"{'Snowball Stemmer':<25} | {snowball_stems[:10]}")
    print(f"{'Lemmatization':<25} | {lemmas[:10]}")

# Create a sample file for demonstration
sample_text = "Stemming reduces words to their root form. Lemmatization ensures roots are proper words. Running, runs, and ran all become run."
with open('sample_text.txt', 'w', encoding='utf-8') as f:
    f.write(sample_text)

# Run the processing function
process_text('sample_text.txt')
