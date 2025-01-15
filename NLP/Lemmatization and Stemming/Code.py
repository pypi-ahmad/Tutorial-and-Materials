import nltk
from nltk.stem import WordNetLemmatizer

# initialize the WordNetLemmatizer object
lemmatizer = WordNetLemmatizer()

# example sentence
sentence = "The dogs are barking outside"

# tokenize the sentence
tokens = nltk.word_tokenize(sentence)

# apply lemmatization to each token
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

print(lemmatized_tokens)


# In the above code, we first import the necessary libraries and initialize the WordNetLemmatizer object. We then tokenize the sentence using nltk.word_tokenize() function and apply lemmatization to each token using lemmatizer.lemmatize() method. Finally, we print the list of lemmatized tokens.

# Similarly, we can perform stemming using NLTK's PorterStemmer or SnowballStemmer classes. Here's an example of how to perform stemming using PorterStemmer:


import nltk
from nltk.stem import PorterStemmer

# initialize the PorterStemmer object
stemmer = PorterStemmer()

# example sentence
sentence = "The dogs are barking outside"

# tokenize the sentence
tokens = nltk.word_tokenize(sentence)

# apply stemming to each token
stemmed_tokens = [stemmer.stem(token) for token in tokens]

print(stemmed_tokens)

#In the above code, we initialize the PorterStemmer object, tokenize the sentence, and apply stemming to each token using stemmer.stem() method. Finally, we print the list of stemmed tokens.
