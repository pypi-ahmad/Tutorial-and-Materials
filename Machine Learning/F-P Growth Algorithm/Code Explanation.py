import pyfpgrowth

# create sample dataset
transactions = [['apple', 'bread', 'cheese'],
                ['apple', 'bread'],
                ['apple', 'banana'],
                ['banana', 'bread'],
                ['banana']]

# find frequent itemsets using FP-Growth algorithm
patterns = pyfpgrowth.find_frequent_patterns(transactions, 2)

# generate association rules from frequent itemsets
rules = pyfpgrowth.generate_association_rules(patterns, 0.7)

print("Frequent Itemsets: ", patterns)
print("Association Rules: ", rules)

# In this code, we first create a sample dataset transactions consisting of lists of items purchased by customers. We then use the pyfpgrowth library to find frequent itemsets in the dataset using the find_frequent_patterns function with a minimum support of 2 (meaning an itemset must appear in at least 2 transactions to be considered frequent).

# We then use the generate_association_rules function to generate association rules from the frequent itemsets with a minimum confidence of 0.7 (meaning a rule must be true at least 70% of the time to be considered strong).

# Finally, we print the frequent itemsets and association rules found by the algorithm.