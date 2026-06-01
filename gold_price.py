import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

from google.colab import drive
drive.mount('/content/drive')

gold_data = pd.read_csv('/content/drive/MyDrive/datasets/archive (3)/gld_price_data.csv')

gold_data.head()
gold_data.tail()

gold_data.shape

gold_data.info()

# Missing values
gold_data.isnull().sum()

# Statistical information
gold_data.describe()

print(gold_data.dtypes)

# Correlation Matrix
correlation = gold_data.drop('Date', axis=1).corr()
print(correlation)

gold_data.columns

numeric_data = gold_data.drop('Date', axis=1)
print(numeric_data.head())

correlation = numeric_data.corr()
print(correlation)

plt.figure(figsize=(8,8))
sns.heatmap(
    correlation,
    cbar=True,
    square=True,
    fmt='.1f',
    annot=True,
    annot_kws={'size':8},
    cmap='Blues'
)

# Correlation values of GLD
print(correlation['GLD'])

# Distribution of Gold Price
sns.distplot(gold_data['GLD'], color='g')

# Features and Target
X = gold_data.drop(['Date', 'GLD'], axis=1)
Y = gold_data['GLD']

print(X)
print(Y)

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=2
)

# Random Forest Regressor
regressor = RandomForestRegressor(n_estimators=100)

# Training
regressor.fit(X_train, Y_train)

# Prediction
test_data_prediction = regressor.predict(X_test)

print(test_data_prediction)

# R2 Score
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R Square Error : ", error_score)

# Actual vs Predicted
Y_test = list(Y_test)

plt.plot(Y_test, color='blue', label='Actual Value')
plt.plot(test_data_prediction, color='green', label='Predicted Value')

plt.legend()
plt.show()
