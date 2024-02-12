# train a multilabel classifier model

# IMPORT STATEMENTS
import evaluate
import numpy as np
import logging
import torch

from transformers import Trainer
from transformers.trainer_callback import PrinterCallback
from datasets import Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from config import train_config as config

# from other modules here
from mavpdl1.utils.utils import (
    condense_df,
    CustomCallback,
    # LoggerCallback
)
from mavpdl1.preprocessing.data_preprocessing import PDL1Data
from mavpdl1.model.classification_model import BERTTextSinglelabelClassifier

# load seqeval in evaluate
seqeval = evaluate.load("seqeval")


def optuna_hp_space(trial):
    """
    Get a hyperparameter space for optuna backend to use with a
    trainer trial
    :param trial: a trainer trial
    :return:
    """
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [1, 2, 4]),
        "num_train_epochs": trial.suggest_categorical("num_train_epochs", [1, 2, 4])
    }


if __name__ == "__main__":
    # PREPARE DATA
    # ---------------------------------------------------------
    # use deidentified data sample
    pdl1 = PDL1Data(
        config.dataset_location,
        model=config.model,
        ner_classes=["unit"],
        classification_classes=["test"],
        classification_type="multilabel",
    )
    all_data = pdl1.data

    # get tokenized data and IOB-2 gold labeled data
    tokenized, gold, sids = pdl1.tokenize_label_data()

    # convert data to train, dev, and test
    # percentage breakdown and code formatting from Kyle code
    (
        X_train_full,
        X_test,
        y_train_full,
        y_test,
        ids_train_full,
        ids_test,
    ) = train_test_split(
        tokenized, gold, sids, test_size=0.15, random_state=config.seed
    )

    # generate the test dataset if needed
    # train and val done below due to nature of the tasks
    test_dataset = Dataset.from_dict({"texts": X_test, "TIUDocumentSID": ids_test})
    test_dataset = test_dataset.map(pdl1.tokenize, batched=True)

    # PART 2
    # USE THE IDENTIFIED PD-L1 INPUTS TO TRAIN A CLASSIFIER TO GET
    # VENDOR AND UNIT INFORMATION FROM THESE SAMPLES, WHEN AVAILABLE
    logging.info("Instantiating model")
    classifier = BERTTextSinglelabelClassifier(config, pdl1.cls_encoder, pdl1.tokenizer)

    # get the labeled data for train and dev partition
    # test partition already generated above
    train_ids, val_ids = train_test_split(
        ids_train_full, test_size=0.25, random_state=config.seed
    )

    # get train data using train_ids
    # set of document SIDs was passed to split earlier
    train_df = all_data[all_data["TIUDocumentSID"].isin(train_ids)]
    train_df = condense_df(train_df, pdl1.cls_encoder, gold_types="test")

    val_df = all_data[all_data["TIUDocumentSID"].isin(val_ids)]
    val_df = condense_df(val_df, pdl1.cls_encoder, gold_types="test")

    # convert to dataset
    # for some reason here we need LABEL instead of LABELS
    train_dataset = Dataset.from_dict(
        {
            "texts": train_df["CANDIDATE"].tolist(),
            "label": train_df["GOLD"].tolist(),
            "TIUDocumentSID": train_df["TIUDocumentSID"].tolist(),
        }
    )
    train_dataset = train_dataset.map(pdl1.tokenize, batched=True)

    val_dataset = Dataset.from_dict(
        {
            "texts": val_df["CANDIDATE"].tolist(),
            "label": val_df["GOLD"].tolist(),
            "TIUDocumentSID": val_df["TIUDocumentSID"].tolist(),
        }
    )
    val_dataset = val_dataset.map(pdl1.tokenize, batched=True)

    # instantiate trainer for hyperparameter search
    classification_trainer = Trainer(
        model_init=classifier.reinit_model,
        args=classifier.training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=classifier.compute_metrics,
    )
    # add callback to print train compute_metrics for train set
    # in addition to val set
    classification_trainer.add_callback(CustomCallback(classification_trainer))
    # classification_trainer.add_callback(LoggerCallback())

    def compute_objective(metrics):
        return metrics['eval_f1']

    # train the model
    logging.info("Beginning training for classification")
    logging.info("Hyperparameter search! ")
    classification_metrics = classification_trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=5,
        compute_objective=compute_objective
    )
    logging.info("Best model parameters from hyperparameter search:")
    logging.info(classification_metrics)
    # classification_metrics = classification_trainer.train()

    # retrain the best model parameters
    # todo: it would be better to load the best model directly
    logging.info("About to retrain model using best hyperparameters")
    classifier.load_best_hyperparameters(classification_metrics.hyperparameters)

    best_cls_trainer = Trainer(
        model=classifier.model,
        args=classifier.training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=classifier.compute_metrics
    )

    trained = best_cls_trainer.train()
    logging.info("Model retrained on best hyperparameters.")

    metrics = trained.metrics

    best_cls_trainer.log_metrics("all", metrics)
    best_cls_trainer.save_metrics("all", metrics)

    logging.info("Results of our model on the validation dataset: ")
    # look at best model performance on validation dataset
    y_pred = best_cls_trainer.predict(val_dataset)
    preds = torch.argmax(torch.Tensor(y_pred.predictions), dim=1)

    labels = val_dataset["label"]

    # get confusion matrix
    logging.info(f"Confusion matrix on validation set: ")
    logging.info(confusion_matrix(labels, preds))

    # get classification report on val set
    logging.info(f"Classification report on validation set: ")
    logging.info(
        classification_report(
            labels,
            preds,
            labels=np.array([i for i in range(len(pdl1.cls_encoder.classes_))]),
            target_names=pdl1.cls_encoder.classes_,
            zero_division=0.0,
        )
    )
