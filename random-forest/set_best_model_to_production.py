import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType

mlflow.set_tracking_uri("http://localhost:5001")

client = MlflowClient()

experiment = mlflow.get_experiment_by_name("random_forest_train")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.mse ASC"]
)

run = runs[0]

new_run_id = run.info.run_id

model = client.search_model_versions("run_id='{}'".format(new_run_id))[0]

version = model.version
name = model.name

client.transition_model_version_stage(name, version, "Production")
