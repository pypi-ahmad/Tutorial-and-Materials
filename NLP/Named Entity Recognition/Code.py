import spacy

# Load English language model in spaCy
nlp = spacy.load("en_core_web_sm")

# Sample text to perform Named Entity Recognition on
text = "Apple is looking at buying a U.K. startup for $1 billion"

# Process text with spaCy
doc = nlp(text)

# Loop through entities in the text and print them along with their labels
for ent in doc.ents:
    print(ent.text, ent.label_)


# In this code, we first load the English language model in spaCy. Then, we define a sample text that we want to perform Named Entity Recognition on. We then process the text with spaCy and obtain a doc object. Finally, we loop through the entities in the text using the .ents property of the doc object, and print each entity along with its label.

# The output of this code for the sample text above would be:


# Apple ORG
# U.K. GPE
# $1 billion MONEY


# Here, ORG refers to the organization named "Apple", GPE refers to the geopolitical entity named "U.K.", and MONEY refers to the monetary value of "$1 billion".