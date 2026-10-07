import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load Titanic training data from a public mirror.
url = "https://raw.githubusercontent.com/dsindy/kaggle-titanic/master/data/train.csv"
titanic = pd.read_csv(url)

# Select passenger fare and three predictors.
df = titanic[["Fare", "Pclass", "Age", "SibSp"]].copy()

# Fill missing ages with the median.
df["Age"] = df["Age"].fillna(df["Age"].median())

X = df[["Pclass", "Age", "SibSp"]]
y = df["Fare"]

# Fit the regression model.
model = LinearRegression()
model.fit(X, y)

print(df.head())
print("Intercept:", model.intercept_)
print("Coefficients:")
print(pd.Series(model.coef_, index=X.columns))
