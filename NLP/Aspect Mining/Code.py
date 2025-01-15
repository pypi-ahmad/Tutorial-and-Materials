from textblob import TextBlob
from collections import defaultdict

def aspect_mining(text):
    # Create TextBlob object
    blob = TextBlob(text)
    # Initialize dictionary to store aspect terms and their frequency
    aspect_terms = defaultdict(int)
    # Get noun phrases
    for np in blob.noun_phrases:
        # Check if np is not a stopword
        if np.lower() not in stop_words:
            # Check if np is a noun
            if np.split()[0].lower() in nouns:
                aspect_terms[np] += 1
    # Sort aspect terms by frequency
    sorted_terms = sorted(aspect_terms.items(), key=lambda x: x[1], reverse=True)
    return sorted_terms


# This code uses the TextBlob library to perform aspect mining on a given text. The aspect_mining function takes in a text parameter and returns a list of aspect terms sorted by their frequency. The function first creates a TextBlob object for the input text. It then loops over all noun phrases in the text and checks if the phrase is a noun and not a stopword. If it is, the phrase is added to the aspect_terms dictionary with a frequency of 1. Finally, the aspect terms are sorted by frequency and returned.

# Note that this code assumes that the stop_words and nouns variables have already been defined. These can be obtained using NLTK or another library.