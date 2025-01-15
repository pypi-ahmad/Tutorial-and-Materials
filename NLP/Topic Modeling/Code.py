# import necessary libraries
import gensim
from gensim import corpora
from pprint import pprint

# sample documents
documents = ["Gardening is a hobby that I enjoy",
             "I have a small garden outside my apartment",
             "Tomatoes and peppers are my favorite vegetables",
             "I enjoy learning about different types of plants",
             "My garden brings me joy and relaxation"]

# preprocess the documents
stopwords = set('for a of the and to in my'.split())
texts = [[word for word in document.lower().split() if word not in stopwords] for document in documents]

# create a dictionary
dictionary = corpora.Dictionary(texts)

# create a corpus
corpus = [dictionary.doc2bow(text) for text in texts]

# train the LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=dictionary,
                                            num_topics=3,
                                            random_state=42,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)

# print the topics and their corresponding words
pprint(lda_model.print_topics())


# In this example, we first import the necessary libraries, including gensim for Topic Modeling. Then, we define a list of sample documents and preprocess them by removing stop words.

# Next, we create a dictionary and a corpus from the preprocessed documents. We then train the LDA model with the gensim.models.ldamodel.LdaModel method, specifying the corpus, dictionary, number of topics, random state, passes, alpha, and per-word topics. Finally, we print the topics and their corresponding words using the print_topics() method.

# This code is a basic example of how to implement Topic Modeling using LDA with the gensim library in Python. It can be further optimized and customized based on specific use cases and requirements.