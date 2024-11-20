import random
import pickle
import os
from collections import Counter, defaultdict

class NameGenerator:
    VOWEL_COUNTER = Counter({'a': 1, 'e': 1, 'i': 1, 'o': 1, 'u': 1})

    def __init__(self, name_file):
        self.name_file = name_file
        self.training_data_file = name_file.replace('.txt', '.pkl')
        self.letter_pairs = defaultdict(Counter)
        self.trigrams = set()
        self.first_letter_distribution = Counter()
        self.length_distribution = Counter()
        self.first_two_letter_digrams = set()
        self.last_two_letter_digrams = set()
        
        # Train the model if the training data file does not exist, otherwise load it
        if not os.path.exists(self.training_data_file):
            self._analyze_names()
            self._save_training_data()
        else:
            self._load_training_data()

    def _analyze_names(self):
        """Load names from a text file and analyze them to extract letter pairs, trigrams, and distributions."""
        with open(self.name_file, 'r') as file:
            names = [line.strip().lower() for line in file if len(line.strip()) >= 2]

        for name in names:
            self.first_letter_distribution[name[0]] += 1
            self.length_distribution[len(name)] += 1

            self.first_two_letter_digrams.add(name[:2])
            self.last_two_letter_digrams.add(name[-2:])

            for i in range(len(name) - 1):
                self.letter_pairs[name[i]][name[i + 1]] += 1
                if i < len(name) - 2:
                    self.trigrams.add(name[i:i + 3])

    def _save_training_data(self):
        """Save the training data to a file."""
        with open(self.training_data_file, 'wb') as file:
            data = (self.letter_pairs, self.length_distribution, self.first_letter_distribution, 
                    self.trigrams, self.first_two_letter_digrams, self.last_two_letter_digrams)
            pickle.dump(data, file)

    def _load_training_data(self):
        """Load the training data from a file."""
        with open(self.training_data_file, 'rb') as file:
            (self.letter_pairs, self.length_distribution, self.first_letter_distribution, 
             self.trigrams, self.first_two_letter_digrams, self.last_two_letter_digrams) = pickle.load(file)

    def _weighted_random_choice(self, distribution):
        """Select a random item from a distribution, weighted by counts."""
        items, weights = zip(*distribution.items())
        return random.choices(items, weights=weights, k=1)[0]

    def _generate_next_char(self, name, length, noise):
        """Generate the next character for the name based on constraints and noise level."""
        counter = 0
        use_random = random.random() < noise
        
        while True:
            counter += 1
            if counter >= 100:
                return random.choice('aeiou')  # Emergency fallback
            
            if use_random:
                next_char = random.choice('abcdefghijklmnopqrstuvwxyz')
            else:
                next_chars = self.letter_pairs.get(name[-1], self.VOWEL_COUNTER)  # fallback to vowels if no pair found
                next_char = self._weighted_random_choice(next_chars)
            
            # Reject if the next character would break the digram or trigram rules
            if len(name) == 1 and name[0] + next_char not in self.first_two_letter_digrams:
                continue
            if len(name) >= 2 and name[-2:] + next_char not in self.trigrams:
                continue
            if len(name) >= length - 1 and name[-1] + next_char not in self.last_two_letter_digrams:
                continue
            
            return next_char

    def random_name(self, noise=0.0):
        """
        Generate a random name based on analyzed distributions and constraints. Will attempt to ensure the first two letters,
        last two letters, and all three-letter pairs are seen in the training set. 
        
        Parameters:
        noise (float): A value between 0.0 and 1.0 that determines the likelihood of choosing a completely random letter
                       instead of following the analyzed distributions, while maintaining digram and trigram rules.
                       Higher values increase randomness.
        """
        while True:
            length = self._weighted_random_choice(self.length_distribution)
            name = self._weighted_random_choice(self.first_letter_distribution)
            
            for _ in range(length - 1):
                name += self._generate_next_char(name, length, noise)
            
            # Reject entire name if no vowel
            if not any(char in 'aeiou' for char in name):
                continue
            
            return name.capitalize()

if __name__ == "__main__":
    european_generator = NameGenerator('european.txt')
    arabic_generator = NameGenerator('arabic.txt')
    indian_generator = NameGenerator('indian.txt')
    hispanic_generator = NameGenerator('hispanic.txt')
    
    headers = ["European", "Arabic", "Indian", "Hispanic"]
    print(f"{headers[0][:15]:<16} {headers[1][:15]:<16} {headers[2][:15]:<16} {headers[3][:15]:<16}")
    print("-" * 64)
    
    noise_value = 0.2
    for _ in range(10):
        european_name = european_generator.random_name(noise=noise_value)
        arabic_name = arabic_generator.random_name(noise=noise_value)
        indian_name = indian_generator.random_name(noise=noise_value)
        hispanic_name = hispanic_generator.random_name(noise=noise_value)
        print(f"{european_name[:15]:<16} {arabic_name[:15]:<16} {indian_name[:15]:<16} {hispanic_name[:15]:<16}")