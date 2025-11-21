import nltk

# Download required resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Read text from file
with open('sample_text.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Tokenize and POS tag
sentences = nltk.sent_tokenize(text)
for sent in sentences:
    words = nltk.word_tokenize(sent)
    tagged = nltk.pos_tag(words)

    # Define a chunk grammar: NP chunk everything, then chink verbs and prepositions
    grammar = r"""
      NP: {<.*>+}          # Chunk everything
          }<VB.*|IN>{      # Chink verbs and prepositions
    """

    chunk_parser = nltk.RegexpParser(grammar)
    chunked = chunk_parser.parse(tagged)

    print(chunked)
    # To visualize, uncomment the next line:
    # chunked.draw()


