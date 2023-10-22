# import mlflow

# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_diabetes
# from sklearn.ensemble import RandomForestRegressor

# mlflow.autolog()

# db = load_diabetes()
# X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# # Create and train models.
# rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
# rf.fit(X_train, y_train)

# # Use the model to make predictions on the test dataset.
# predictions = rf.predict(X_test)


from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

import time
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

mlflow.set_tracking_uri("http://localhost:5001")

mlflow.set_experiment("random_forest_train")
experiment = mlflow.get_experiment_by_name("random_forest_train")

with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
    X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    params = {"max_depth": 40 , "random_state": 42}
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    # Infer the model signature
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()
    signature = infer_signature(X_test, y_pred)

    # Defining RMSE function
    def root_mean_squared_error(actual, predictions):
        return np.sqrt(mean_squared_error(actual, predictions))

    def time_predict(start, end):
        return (end - start)

    # Log parameters and metrics using the MLflow APIs
    mlflow.log_params(params)
    mlflow.log_metrics({"mse": mean_squared_error(y_test, y_pred)})
    mlflow.log_metrics({"r2": r2_score(y_test, y_pred)})
    mlflow.log_metrics({"rmse": root_mean_squared_error(y_test, y_pred)})
    mlflow.log_metrics({"prediction time for test data": time_predict(start_time, end_time)})

    # Log the sklearn model and register as version 1
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="sklearn-model",
        signature=signature,
        registered_model_name="sk-learn-random-forest-reg-model",
    )
