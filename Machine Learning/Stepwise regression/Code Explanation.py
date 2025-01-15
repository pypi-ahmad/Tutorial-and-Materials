import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load Boston Housing dataset
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target, name='MEDV')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize selected features and p-value threshold
selected = []
p_threshold = 0.05

# Stepwise regression algorithm
while len(selected) < len(boston.feature_names):
    # Initialize best p-value and feature
    best_pval = 1
    best_feature = None
    
    # Iterate over features not yet selected
    for feature in boston.feature_names:
        if feature not in selected:
            # Add feature to selected features
            X_train_sel = X_train[selected + [feature]]
            
            # Fit linear regression model and calculate p-value
            lr = LinearRegression().fit(X_train_sel, y_train)
            y_pred = lr.predict(X_train_sel)
            mse = mean_squared_error(y_train, y_pred)
            n = len(X_train_sel)
            p = len(selected) + 1
            F = ((mse_null - mse) / p) / (mse / (n - p - 1))
            pval = 1 - stats.f.cdf(F, p, n - p - 1)
            
            # Update best feature if p-value is lower than current best
            if pval < best_pval:
                best_pval = pval
                best_feature = feature
    
    # If best p-value is below threshold, add best feature to selected features
    if best_pval < p_threshold:
        selected.append(best_feature)
    else:
        break

# Print selected features
print("Selected Features:", selected)
