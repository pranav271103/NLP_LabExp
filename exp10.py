import spacy
import argparse
from pathlib import Path

def load_model():
    """Load the English language model for NER"""
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        print("Downloading the English language model...")
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

def read_text_from_file(file_path):
    """Read text from a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def perform_ner(text, nlp):
    """Perform Named Entity Recognition on the given text"""
    doc = nlp(text)
    
    # Extract named entities
    entities = []
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'start': ent.start_char,
            'end': ent.end_char,
            'label': ent.label_
        })
    
    return entities, doc

def display_entities(entities):
    """Display the named entities in a formatted way"""
    if not entities:
        print("No named entities found in the text.")
        return
    
    print("\nFound the following named entities:")
    print("-" * 60)
    print(f"{'Text':<30} {'Type':<20} {'Position'}")
    print("-" * 60)
    
    for ent in entities:
        print(f"{ent['text']:<30} {ent['label']:<20} ({ent['start']}-{ent['end']})")

def highlight_entities(text, doc):
    """Return the text with entities highlighted"""
    from termcolor import colored
    
    if not doc.ents:
        return text
    
    # Sort entities by start position in reverse order to handle replacements correctly
    sorted_entities = sorted(doc.ents, key=lambda x: x.start_char, reverse=True)
    
    # Convert to list for manipulation
    text_list = list(text)
    
    for ent in sorted_entities:
        start = ent.start_char
        end = ent.end_char
        entity_text = ent.text
        label = ent.label_
        
        # Create highlighted version
        highlighted = colored(entity_text, 'red') + colored(f'({label})', 'blue')
        
        # Replace the original text with highlighted version
        text_list[start:end] = highlighted
    
    return ''.join(text_list)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Perform Named Entity Recognition on a text file.')
    parser.add_argument('file_path', type=str, help='Path to the input text file')
    args = parser.parse_args()
    
    # Check if file exists
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"Error: File '{file_path}' does not exist.")
        return
    
    # Load the NLP model
    print("Loading NLP model...")
    nlp = load_model()
    
    # Read the input file
    print(f"Reading file: {file_path}")
    text = read_text_from_file(file_path)
    if text is None:
        return
    
    print("\nOriginal Text:")
    print("-" * 60)
    print(text[:500] + ("..." if len(text) > 500 else ""))
    
    # Perform NER
    print("\nPerforming Named Entity Recognition...")
    entities, doc = perform_ner(text, nlp)
    
    # Display results
    display_entities(entities)
    
    # Show text with highlighted entities (if termcolor is available)
    try:
        from termcolor import colored
        print("\nText with entities highlighted:")
        print("-" * 60)
        print(highlight_entities(text, doc)[:1000] + ("..." if len(text) > 1000 else ""))
    except ImportError:
        print("\nInstall 'termcolor' package to see highlighted entities. Run: pip install termcolor")

if __name__ == "__main__":
    main()
