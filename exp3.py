import nltk
import re
from nltk.corpus import brown

def find_instances_of_the():
    # Download the Brown corpus if not already downloaded
    try:
        nltk.data.find('corpora/brown')
    except LookupError:
        nltk.download('brown')
    
    # Get all words from the Brown corpus
    words = brown.words()
    
    # Use regex to find all instances of 'the' (case insensitive)
    # \b ensures we match whole words only
    the_pattern = re.compile(r'\bthe\b', re.IGNORECASE)
    
    # Find all matches and their positions
    matches = []
    for i, word in enumerate(words):
        if the_pattern.fullmatch(word):
            # Store the word and its position
            matches.append((i, word))
    
    # Print results
    print(f"Found {len(matches)} instances of 'the' in the Brown corpus.")
    print("\nFirst 20 instances:")
    for i, (pos, word) in enumerate(matches[:20], 1):
        print(f"{i}. Position {pos}: '{word}'")
    
    # Optional: Save all matches to a file
    with open('the_instances.txt', 'w', encoding='utf-8') as f:
        for pos, word in matches:
            f.write(f"{pos}\t{word}\n")
    print("\nAll instances have been saved to 'the_instances.txt'")

if __name__ == "__main__":
    find_instances_of_the()


