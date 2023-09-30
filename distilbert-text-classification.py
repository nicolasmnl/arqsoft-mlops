import sys, warnings

import ktrain
from ktrain import text
from sklearn import model_selection
import pandas as pd

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    try:
        df = pd.read_csv("data/text-classification.csv")
        print("Number of samples for each class: ")
        print(df["label"].value_counts())
        classes = list(set(df.label.tolist()))
    except Exception as e:
        logger.exception(
            "Unable to load training & test CSV, check the file path. Error: %s", e
        )

    X_train, x_test, y_train, y_test = model_selection.train_test_split(df['content'].tolist(), df['label'].tolist())
    trn, val, preproc = text.texts_from_array(x_train=X_train, y_train=y_train,
                                                x_test=x_test, y_test=y_test,
                                                class_names=classes,
                                                preprocess_mode='distilbert',
                                                maxlen=256, 
                                                max_features=10000)
    model = text.text_classifier('distilbert', train_data=trn, preproc=preproc)
    learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=6, use_multiprocessing=True)

    # learner.lr_find(max_epochs=2) # finding the learning rate
    # learner.lr_plot()

    lrate = 2e-5
    epochs = 1
    print("Training ktrain Distilbert model (lrate={:f}, epochs={:f}):".format(lrate, epochs))
    mlflow.set_experiment("text_classification_ktrain")
    experiment = mlflow.get_experiment_by_name("text_classification_ktrain")
    with mlflow.start_run(experiment_id=experiment.experiment_id, nested=True):
        learner.fit_onecycle(lrate, epochs)
        predictor = ktrain.get_predictor(learner.model, preproc)
        y_pred = predictor.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        mlflow.log_metric("accuracy", round(accuracy, 4))
        mlflow.log_metric("precision", round(precision, 4))
        mlflow.log_metric("recall", round(recall, 4))
        mlflow.log_metric("f1-score", round(f1, 4))
        mlflow.log_param("lrate", lrate)
        mlflow.log_param("epochs", epochs)
        signature = infer_signature(x_test, y_pred)
        # save the model
        mlflow.sklearn.log_model(
            sk_model=learner,
            signature=signature,
            artifact_path="text_classification1",
            registered_model_name="distilbert-text-classification-model",
        )
