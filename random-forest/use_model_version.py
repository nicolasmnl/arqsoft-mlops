import mlflow.pyfunc

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

model_name = "sk-learn-random-forest-reg-model"
model_version = 1

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

predictions = model.predict(X_test)
print(predictions)
