import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Load the dataset
dataset = pd.read_csv('groceries.csv', header=None)

# Convert the dataset into a list of lists
transactions = []
for i in range(0, len(dataset)):
    transactions.append([str(dataset.values[i,j]) for j in range(0, len(dataset.columns))])

# Apply Apriori algorithm
frequent_itemsets = apriori(transactions, min_support=0.01, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Display the top 5 rules
print(rules.head())

# In this code, we first load the grocery dataset and convert it into a list of lists where each inner list represents a transaction. We then apply the Apriori algorithm on this list of transactions to find the frequent itemsets that meet the minimum support threshold. Finally, we generate the association rules based on the frequent itemsets and a given minimum lift threshold. The code outputs the top 5 rules based on the lift metric. The mlxtend library provides an easy-to-use implementation of the Apriori algorithm in Python.