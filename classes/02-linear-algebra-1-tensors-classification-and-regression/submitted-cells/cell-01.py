#Linear Regression here
#trained on 100, tested on 50

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

iris = load_iris()

# Inputs: sepal width, petal length, petal width, species.
X = np.column_stack([iris.data[:, 1:], iris.target])

# Target: sepal length.
y = iris.data[:, 0]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=101
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

predictions = linear_model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, predictions))
print("MSE:", mean_squared_error(y_test, predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))

# Multiple inputs: compare actual and predicted sepal lengths.
plt.scatter(y_test, predictions)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)
plt.xlabel("Actual sepal length (cm)")
plt.ylabel("Predicted sepal length (cm)")
plt.title("Linear regression")
plt.show()
