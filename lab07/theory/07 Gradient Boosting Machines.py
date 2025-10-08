# Import necessary libraries for classification and regression
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# Load Iris dataset for classification
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

# Load California housing dataset for regression
california_housing = fetch_california_housing()
X_california, y_california = california_housing.data, california_housing.target
X_train_california, X_test_california, y_train_california, y_test_california = train_test_split(X_california, y_california, test_size=0.2, random_state=42)

# Gradient Boosting for Classification (Iris dataset)
gbc = GradientBoostingClassifier(random_state=42)
gbc.fit(X_train_iris, y_train_iris)
y_pred_iris = gbc.predict(X_test_iris)
accuracy_iris = accuracy_score(y_test_iris, y_pred_iris)

# Gradient Boosting for Regression (California housing dataset)
gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(X_train_california, y_train_california)
y_pred_california = gbr.predict(X_test_california)
mse_california = mean_squared_error(y_test_california, y_pred_california)

accuracy_iris, mse_california
