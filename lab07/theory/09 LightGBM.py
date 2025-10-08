#!pip install lightgbm

# Importar las bibliotecas necesarias
import lightgbm as lgb
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Cargar el dataset Iris para la clasificación
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

# Cargar el dataset California Housing para la regresión
california_housing = fetch_california_housing()
X_california, y_california = california_housing.data, california_housing.target
X_train_california, X_test_california, y_train_california, y_test_california = train_test_split(X_california, y_california, test_size=0.2, random_state=42)

# LightGBM para la clasificación (Iris)
lgb_clf = lgb.LGBMClassifier(random_state=42)
lgb_clf.fit(X_train_iris, y_train_iris)
y_pred_iris_lgb = lgb_clf.predict(X_test_iris)
accuracy_iris_lgb = accuracy_score(y_test_iris, y_pred_iris_lgb)

# LightGBM para la regresión (California Housing)
lgb_reg = lgb.LGBMRegressor(random_state=42)
lgb_reg.fit(X_train_california, y_train_california)
y_pred_california_lgb = lgb_reg.predict(X_test_california)
mse_california_lgb = mean_squared_error(y_test_california, y_pred_california_lgb)

# Imprimir resultados
print(f"Precisión en clasificación (Iris): {accuracy_iris_lgb}")
print(f"Error cuadrático medio en regresión (California Housing): {mse_california_lgb}")
