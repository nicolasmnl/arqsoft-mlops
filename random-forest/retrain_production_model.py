import mlflow
from mlflow import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

import time

mlflow.set_tracking_uri("http://localhost:5001")

client = MlflowClient()

X, y = make_regression(n_samples=400, n_features=4, n_informative=2, random_state=0, shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_uri = 'models:/sk-learn-random-forest-reg-model/Production'
model = mlflow.sklearn.load_model(model_uri)

prod_run_id = client.get_latest_versions("sk-learn-random-forest-reg-model", stages=["Production"])[0].run_id

params = client.get_run(prod_run_id).data.params

y_pred = model.predict(X_test)

prod_mse = mean_squared_error(y_test, y_pred)

experiment = mlflow.get_experiment_by_name("random_forest_train")

with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    new_mse = mean_squared_error(y_test, y_pred)

    # Defining RMSE function
    def root_mean_squared_error(actual, predictions):
        return np.sqrt(mean_squared_error(actual, predictions))

    mlflow.log_params(params)
    mlflow.log_metrics({"mse": mean_squared_error(y_test, y_pred)})
    mlflow.log_metrics({"r2": r2_score(y_test, y_pred)})
    mlflow.log_metrics({"rmse": root_mean_squared_error(y_test, y_pred)})

new_run_id = run.info.run_id

print(new_mse, prod_mse)

if (new_mse < prod_mse):
    name = "sk-learn-random-forest-reg-model"
    # register new model version
    model_uri = f"runs:/{new_run_id}/sklearn-model"
    mv = client.create_model_version(name, model_uri, new_run_id)

    to_prod_version = client.search_model_versions("run_id='{}'".format(new_run_id))[0].version
    to_archive_version = client.search_model_versions("run_id='{}'".format(prod_run_id))[0].version

    # Transition new model to Production stage
    client.transition_model_version_stage(name, to_prod_version, "Production")

    # Wait for the transition to complete
    new_prod_version = client.get_model_version(name, to_prod_version)
    while new_prod_version.current_stage != "Production":
        new_prod_version = client.get_model_version(name, to_prod_version)
        print('Transitioning new model... Current model version is: ', new_prod_version.current_stage)
        time.sleep(1)

    # Transition old model to Archived stage
    client.transition_model_version_stage(name, to_archive_version, "Archived")

else:
    print('no improvement')

