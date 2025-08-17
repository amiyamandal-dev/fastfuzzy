#!/usr/bin/env python3
"""
Spell Checker with FastFuzzy

This script demonstrates how to build a spell checker using the fastfuzzy library.
"""

import fastfuzzy
import re
import time
from typing import List, Tuple, Dict, Set

class SpellChecker:
    """A spell checker using fuzzy string matching."""
    
    def __init__(self, dictionary: List[str], threshold: float = 80.0):
        self.dictionary = dictionary
        self.threshold = threshold
        self.process = fastfuzzy.process
        self.fuzzy = fastfuzzy.FuzzyRatio()
        
        # Preprocess dictionary for faster lookup
        self.processed_dict = {word.lower(): word for word in dictionary}
        self.dict_keys = list(self.processed_dict.keys())
        
        print(f"SpellChecker initialized with {len(dictionary)} words")
    
    def is_word_correct(self, word: str) -> bool:
        """Check if a word is spelled correctly."""
        return word.lower() in self.processed_dict
    
    def get_corrections(self, word: str, limit: int = 5) -> List[Tuple[str, float]]:
        """Get spelling corrections for a word."""
        if self.is_word_correct(word):
            return [(word, 100.0)]
        
        # Use fuzzy search to find similar words
        results = self.process.extract(
            word.lower(), 
            self.dict_keys, 
            limit=limit,
            score_cutoff=self.threshold,
            scorer="ratio"
        )
        
        # Map back to original case
        corrections = []
        for dict_word_key, score, index in results:
            original_word = self.processed_dict[dict_word_key]
            corrections.append((original_word, score))
        
        return corrections
    
    def check_text(self, text: str) -> Dict[str, List[Tuple[str, float]]]:
        """Check spelling in a text and return corrections."""
        # Extract words from text
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        
        # Check each word
        misspellings = {}
        for word in words:
            if not self.is_word_correct(word):
                corrections = self.get_corrections(word, limit=3)
                if corrections:
                    misspellings[word] = corrections
        
        return misspellings
    
    def suggest_sentence_corrections(self, sentence: str, max_suggestions: int = 3) -> List[str]:
        """Suggest corrected versions of a sentence."""
        words = sentence.split()
        corrected_words = []
        
        for word in words:
            # Remove punctuation for checking
            clean_word = re.sub(r'[^\\w]', '', word)
            punctuation = re.sub(r'[\\w]', '', word)
            
            if clean_word and not self.is_word_correct(clean_word):
                corrections = self.get_corrections(clean_word, limit=1)
                if corrections:
                    corrected_word = corrections[0][0] + punctuation
                    corrected_words.append(corrected_word)
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
        
        return [' '.join(corrected_words)]

def load_dictionary(file_path: str = None) -> List[str]:
    """Load dictionary words."""
    # If no file provided, use a basic English word list
    if not file_path:
        # Basic English words for demonstration
        basic_words = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their",
            "what", "so", "up", "out", "if", "about", "who", "get", "which", "go",
            "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
            "take", "people", "into", "year", "your", "good", "some", "could", "them",
            "see", "other", "than", "then", "now", "look", "only", "come", "its",
            "over", "think", "also", "back", "after", "use", "two", "how", "our",
            "work", "first", "well", "way", "even", "new", "want", "because", "any",
            "these", "give", "day", "most", "us", "is", "was", "are", "has", "had",
            "been", "were", "being", "have", "do", "does", "did", "shall", "will",
            "should", "would", "may", "might", "must", "can", "could", "ought",
            "apple", "banana", "computer", "programming", "python", "language",
            "algorithm", "function", "variable", "string", "integer", "boolean",
            "array", "list", "dictionary", "class", "object", "method", "library",
            "framework", "database", "server", "client", "network", "internet",
            "website", "application", "software", "hardware", "memory", "processor",
            "keyboard", "mouse", "screen", "monitor", "printer", "scanner",
            "machine", "learning", "artificial", "intelligence", "data", "science",
            "analysis", "visualization", "statistics", "mathematics", "physics",
            "chemistry", "biology", "medicine", "engineering", "architecture",
            "design", "graphic", "user", "interface", "experience", "testing",
            "debugging", "optimization", "performance", "security", "privacy",
            "encryption", "decryption", "authentication", "authorization",
            "blockchain", "cryptocurrency", "bitcoin", "ethereum", "smart",
            "contract", "distributed", "system", "cloud", "storage", "backup",
            "recovery", "disaster", "virtualization", "container", "docker",
            "kubernetes", "microservice", "api", "rest", "graphql", "json",
            "xml", "html", "css", "javascript", "typescript", "react", "angular",
            "vue", "node", "express", "django", "flask", "spring", "java", "c",
            "cpp", "rust", "go", "swift", "kotlin", "scala", "ruby", "php",
            "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
            "kafka", "rabbitmq", "nginx", "apache", "tomcat", "jenkins",
            "git", "github", "gitlab", "bitbucket", "devops", "agile",
            "scrum", "kanban", "waterfall", "documentation", "comment",
            "readme", "license", "copyright", "patent", "trademark",
            "open", "source", "free", "software", "license", "mit", "apache",
            "gpl", "bsd", "creative", "commons", "public", "domain"
        ]
        return basic_words
    
    # If file provided, load from file
    try:
        with open(file_path, 'r') as f:
            words = [line.strip().lower() for line in f if line.strip()]
        return words
    except FileNotFoundError:
        print(f"Dictionary file {file_path} not found, using basic dictionary")
        return load_dictionary()  # Use basic dictionary

def demo_spell_checker():
    """Demonstrate spell checker functionality."""
    print("=== Spell Checker Demo ===")
    
    # Load dictionary
    dictionary = load_dictionary()
    spell_checker = SpellChecker(dictionary, threshold=75.0)
    
    # Test individual word corrections
    test_words = ["pythom", "progaming", "algoritm", "functiom", "vaiable"]
    
    print("\nWord Corrections:")
    for word in test_words:
        corrections = spell_checker.get_corrections(word, limit=3)
        print(f"  '{word}' -> {corrections}")
    
    # Test text checking
    test_texts = [
        "This is a pythom progaming example.",
        "The algoritm uses a functiom to process vaiable data.",
        "I love progaming in pythom language."
    ]
    
    print("\nText Spell Checking:")
    for text in test_texts:
        print(f"\nText: {text}")
        misspellings = spell_checker.check_text(text)
        if misspellings:
            for word, corrections in misspellings.items():
                print(f"  '{word}' suggestions: {corrections}")
        else:
            print("  No spelling errors found")
    
    # Test sentence suggestions
    print("\nSentence Corrections:")
    for text in test_texts:
        suggestions = spell_checker.suggest_sentence_corrections(text)
        print(f"  Original: {text}")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  Suggestion {i}: {suggestion}")

def demo_performance():
    """Demonstrate performance benefits."""
    print("\n=== Performance Demo ===")
    
    # Load a larger dictionary
    dictionary = load_dictionary() * 10  # Make it larger for testing
    spell_checker = SpellChecker(dictionary, threshold=70.0)
    
    # Test words
    test_words = ["pythom", "progaming", "algoritm", "functiom", "vaiable"] * 10
    
    print(f"Testing with {len(test_words)} words against {len(dictionary)} dictionary words")
    
    # Time the operation
    start_time = time.perf_counter()
    for word in test_words:
        spell_checker.get_corrections(word, limit=3)
    end_time = time.perf_counter()
    
    print(f"Completed in {end_time - start_time:.4f} seconds")
    print(f"Average time per word: {(end_time - start_time) / len(test_words) * 1000:.2f} ms")

def demo_advanced_features():
    """Demonstrate advanced spell checker features."""
    print("\n=== Advanced Features Demo ===")
    
    # Create a specialized dictionary
    tech_dictionary = [
        "python", "java", "javascript", "typescript", "rust", "go", "swift",
        "kotlin", "scala", "ruby", "php", "c", "cpp", "csharp", "r",
        "algorithm", "data", "structure", "database", "server", "client",
        "api", "rest", "graphql", "json", "xml", "html", "css",
        "react", "angular", "vue", "node", "express", "django", "flask",
        "spring", "docker", "kubernetes", "microservice", "cloud",
        "aws", "azure", "gcp", "devops", "agile", "scrum", "kanban",
        "git", "github", "gitlab", "bitbucket", "ci", "cd", "jenkins",
        "machine", "learning", "artificial", "intelligence", "neural",
        "network", "deep", "learning", "tensorflow", "pytorch", "scikit",
        "pandas", "numpy", "matplotlib", "seaborn", "plotly"
    ]
    
    tech_spell_checker = SpellChecker(tech_dictionary, threshold=70.0)
    
    # Test technical terms
    tech_words = ["pythom", "javascrpit", "tensrflow", "pytorc", "sciket"]
    
    print("Technical Term Corrections:")
    for word in tech_words:
        corrections = tech_spell_checker.get_corrections(word, limit=3)
        print(f"  '{word}' -> {corrections}")
    
    # Domain-specific text
    tech_text = "I am learning pythom and javascrpit for machne learning with tensrflow."
    print(f"\nTechnical Text: {tech_text}")
    
    misspellings = tech_spell_checker.check_text(tech_text)
    for word, corrections in misspellings.items():
        print(f"  '{word}' -> {corrections}")

def main():
    """Run spell checker demos."""
    print("Spell Checker with FastFuzzy")
    print("=" * 50)
    
    try:
        demo_spell_checker()
        demo_performance()
        demo_advanced_features()
        
        print("\n" + "=" * 50)
        print("Spell checker demo completed!")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()