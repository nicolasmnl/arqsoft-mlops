import mlflow

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

model = mlflow.sklearn.load_model("./mlruns/0/14a6415913044b18bffe159858770f16/artifacts/model")
predictions = model.predict(X_test)
print(predictions)