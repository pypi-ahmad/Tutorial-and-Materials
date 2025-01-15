# import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# create a dataframe with the data
df = pd.DataFrame({
    'Hours Studied': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Exam Grade': [65, 67, 72, 75, 80, 83, 85, 89, 93]
})

# create a scatter plot of the data to visualize the relationship between hours studied and exam grade
plt.scatter(df['Hours Studied'], df['Exam Grade'])
plt.xlabel('Hours Studied')
plt.ylabel('Exam Grade')
plt.show()

# create the linear regression model
model = LinearRegression()

# fit the model to the data
model.fit(df[['Hours Studied']], df['Exam Grade'])

# predict the exam grade for a student who studied for 8 hours
predicted_grade = model.predict([[8]])

# print the predicted grade
print('Predicted grade:', predicted_grade[0])
