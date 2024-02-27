# train an NER model, using hyperparameter tuning on each stage of the model

# IMPORT STATEMENTS
import evaluate
import numpy as np
import logging
import torch
import random

from transformers import (
    Trainer,
    DataCollatorForTokenClassification,
)
from datasets import Dataset

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report
from seqeval.metrics import classification_report as ner_classification_report

from config import train_config as config

# from other modules here
from mavpdl1.utils.utils import (
    id_labeled_items,
    get_from_indexes,
    condense_df,
    CustomCallback,
    CustomCallbackNER,
    id_labeled_items,
    calc_best_hyperparams
)
from mavpdl1.preprocessing.data_preprocessing import PDL1Data
from mavpdl1.model.ner_model import BERTNER
from mavpdl1.model.classification_model import BERTTextSinglelabelClassifier

# load seqeval in evaluate
seqeval = evaluate.load("seqeval")


def compute_objective(metrics):
    return metrics['eval_f1']


# set optuna hyperparameter space
def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", config.cls_lr_min, config.cls_lr_max, log=True),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size",
            config.cls_per_device_train_batch_size
        ),
        "num_train_epochs": trial.suggest_categorical("num_train_epochs", config.cls_num_epochs)}


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
    tokenized, gold = pdl1.tokenize_label_data()

    # convert data to train, dev, and test
    # percentage breakdown and code formatting from Kyle code
    (
        X_train_full,
        X_test,
        y_train_full,
        y_test,
    ) = train_test_split(
        tokenized, gold, test_size=0.15, random_state=config.seed
    )

    # generate the test dataset if needed
    # train and val done below due to nature of the tasks
    test_dataset = Dataset.from_dict({"texts": X_test, "labels": y_test})
    test_dataset = test_dataset.map(pdl1.tokenize, batched=True)

    # convert train data into KFold splits
    splits = config.ner_num_splits
    folds = KFold(n_splits=splits, random_state=config.seed, shuffle=True)
    idxs = range(len(X_train_full))

    # create a holder for all items that you've IDed as
    # having a PD-L1 value
    all_labeled = []

    # instantiate the model and data collator
    logging.info("Instantiating model")
    ner = BERTNER(config, pdl1.ner_encoder, pdl1.tokenizer)

    # add data collator
    data_collator = DataCollatorForTokenClassification(
        pdl1.tokenizer, padding=True, return_tensors="pt"
    )

    # save hyperparameters from first inner trial to test on subsequent ones
    # this will end up as a list of dicts
    # each dict will be in the format {param0: value, param1: value, ...}
    used_hyperparameters = []

    # create a holder for the best hyperparameters for each outer CV loop
    outer_best_hyperparams = []

    # do CV over the data
    for i, (train_index, test_index) in enumerate(folds.split(idxs)):
        logging.info(f"Now beginning fold {i}")

        X_train, X_val = get_from_indexes(X_train_full, train_index, test_index)
        y_train, y_val = get_from_indexes(y_train_full, train_index, test_index)

        # do ANOTHER KFOLD loop nested inside of this for training
        inner_folds = KFold(n_splits=config.ner_num_splits, random_state=config.seed, shuffle=True)
        inner_idxs = range(len(X_train))

        # set datasets for the outer loop
        train_dataset_outer = Dataset.from_dict(
            {"texts": X_train, "labels": y_train}
        )
        train_dataset_outer = train_dataset_outer.map(pdl1.tokenize, batched=True)

        val_dataset_outer = Dataset.from_dict(
            {"texts": X_val, "labels": y_val}
        )
        val_dataset_outer = val_dataset_outer.map(pdl1.tokenize, batched=True)

        best_f1 = -1.0

        # holder for best hyperparameters in the inner loop
        inner_best_hyperparameters = []

        # run hyperparameter selection over the inner folds
        # this will be done using manually-created random search
        for k in range(config.num_trials_in_hyperparameter_search):
            logging.info(f"Now beginning hyperparameter search trial {k} on inner folds for fold {i}")

            if i == 0:
                # generate hyperparameters randomly
                # todo: abstract these numbers out
                new_hyperparams = {
                    "learning_rate": random.uniform(1e-6, 1e-4),
                    "per_device_train_batch_size": random.choice([1, 2, 4]),
                    "num_train_epochs": random.choice([1, 2, 4])
                }

                # add these to the used_hyperparameters holder
                used_hyperparameters.append(new_hyperparams)

            else:
                print(used_hyperparameters)
                # get hyperparameters from used_hyperparameters
                new_hyperparams = used_hyperparameters[k]

            logging.info(f"Hyperparameters trial {k} for split {i}:")
            logging.info(new_hyperparams)

            # add new hyperparameters to model training args
            ner.load_new_hyperparameters(new_hyperparams)

            # holders for predictions and y labels
            # used to calculate performance for each set of hyperparameters
            all_preds = []
            all_ys = []

            # create inner loop for nested CV
            for j, (inner_train_idx, inner_test_idx) in enumerate(inner_folds.split(inner_idxs)):
                logging.info(f"Beginning hyperparameter tuning on INNER fold {j}")

                X_train_inner, X_val_inner = get_from_indexes(X_train, inner_train_idx, inner_test_idx)
                y_train_inner, y_val_inner = get_from_indexes(y_train, inner_train_idx, inner_test_idx)

                # todo: kyle code used `label` but on CPU would not train
                #   unless i changed it to `labels`
                train_dataset_inner = Dataset.from_dict(
                    {"texts": X_train_inner, "labels": y_train_inner}
                )
                train_dataset_inner = train_dataset_inner.map(pdl1.tokenize, batched=True)

                val_dataset_inner = Dataset.from_dict(
                    {"texts": X_val_inner, "labels": y_val_inner}
                )
                val_dataset_inner = val_dataset_inner.map(pdl1.tokenize, batched=True)

                #   and use those in the classification task
                ner.update_save_path(f"{config.savepath}/ner/fold_{i}_inner{j}_trial{k}")
                ner.update_log_path(f"{config.ner_logging_dir}/fold_{i}_inner{j}_trial{k}")

                # reinit the model
                ner.reinit_model()

                # set up trainer
                trainer = Trainer(
                    model=ner.model,
                    args=ner.training_args,
                    train_dataset=train_dataset_inner,
                    data_collator=data_collator,
                    eval_dataset=val_dataset_inner,
                    compute_metrics=ner.compute_metrics,
                )
                # add callback to print train compute_metrics for train set
                # in addition to val set
                trainer.add_callback(CustomCallbackNER(trainer))

                logging.info("Beginning training loop")
                inner_trained = trainer.train()

                # get preds on inner loop
                preds = trainer.predict(val_dataset_inner).predictions
                all_preds.extend(preds)
                all_ys.extend(val_dataset_inner["labels"])

                del trainer

            # get predictions on all partitions of inner loop
            # used to determine which hyperparameters are best
            metrics = ner.compute_metrics([np.asarray(all_preds), np.asarray(all_ys)])
            logging.info("Results of inner loop hyperparameter search for these hyperparameters:")
            logging.info(metrics)
            if metrics["f1"] > best_f1:
                ner.save_best_hyperparameters(new_hyperparams)
                best_f1 = metrics["f1"]

        logging.info("======================================")
        logging.info(f"Best hyperparameters on inner loop for fold {i}")
        logging.info(ner.best_hyperparameters)
        logging.info("======================================")
        outer_best_hyperparams.append(ner.best_hyperparameters)

        # retrain best hyperparameters on OUTER loop
        #   and use those in the classification task
        ner.update_save_path(f"{config.savepath}/ner/fold_{i}_bestparams")
        ner.update_log_path(f"{config.ner_logging_dir}/fold_{i}_bestparams")

        # reinit the model
        ner.reinit_model()

        # set up trainer
        trainer = Trainer(
            model=ner.model,
            args=ner.training_args,
            train_dataset=train_dataset_outer,
            data_collator=data_collator,
            eval_dataset=val_dataset_outer,
            compute_metrics=ner.compute_metrics,
        )
        # add callback to print train compute_metrics for train set
        # in addition to val set
        trainer.add_callback(CustomCallbackNER(trainer))

        logging.info("Beginning training loop")
        inner_trained = trainer.train()

        # get predictions on val dataset from outer loop
        y_pred = trainer.predict(val_dataset_outer)
        labels = val_dataset_outer["labels"]
        metrics = ner.compute_metrics([y_pred.predictions, labels])

        logging.info("Results of inner loop best hyperparameters on outer loop holdout")
        logging.info(metrics)

        # save the best hyperparameters from this inner loop
        outer_best_hyperparams.append(ner.best_hyperparameters)

    # compare the best hyperparameters for each outer loop
    # make a selection on which ones to use for retraining / prediction
    overall_best_hyperparams = calc_best_hyperparams(outer_best_hyperparams)

    ner.load_best_hyperparameters(overall_best_hyperparams)
    logging.info("Overall best hyperparameters calculated from nested CV:")
    logging.info(overall_best_hyperparams)

    logging.info("Now beginning retraining of CV to generate predictions for use with classifier training")
    # rerun CV training on best hyperparameters to get predictions of all items with PDL1 values
    for i, (train_idx, test_idx) in enumerate(folds.split(idxs)):
        logging.info(f"Now beginning final training/prediction on fold {i}")

        # get data and make datasets
        X_train, X_val = get_from_indexes(X_train_full, train_idx, test_idx)
        y_train, y_val = get_from_indexes(y_train_full, train_idx, test_idx)

        train_dataset = Dataset.from_dict(
            {"texts": X_train, "labels": y_train}
        )
        train_dataset = train_dataset.map(pdl1.tokenize, batched=True)

        val_dataset = Dataset.from_dict(
            {"texts": X_val, "labels": y_val}
        )
        val_dataset = val_dataset.map(pdl1.tokenize, batched=True)

        logging.info("Creating trainer")
        # rerun training on best hyperparameters
        best_ner_trainer = Trainer(
            model=ner.model,
            args=ner.training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=ner.compute_metrics
        )

        # train the model
        logging.info("Beginning training loop")
        metrics = best_ner_trainer.train()

        # get the best predictions from the model
        # to select data inputs for classification task
        logging.info("Training loop complete")
        logging.info("Making predictions on held out data")
        y_pred = best_ner_trainer.predict(val_dataset)

        # IDENTIFY ALL SAMPLES THAT WILL MOVE ON TO BE USED AS INPUT WITH NEXT MODEL
        # define a second computation
        logging.info(
            f"Identifying all data points with PDL1 values in validation set for fold {i}:"
        )
        all_labeled_ids = id_labeled_items(
            y_pred.predictions,
            val_dataset["texts"],
            pdl1.ner_encoder.transform(["O"]),
        )
        all_labeled.extend(all_labeled_ids)

    # print(all_labeled)
    logging.info("CV WITH NER MODEL COMPLETE. NOW BEGINNING CLASSIFICATION TASK")
    # PART 2
    # USE THE IDENTIFIED PD-L1 INPUTS TO TRAIN A CLASSIFIER TO GET
    # VENDOR AND UNIT INFORMATION FROM THESE SAMPLES, WHEN AVAILABLE
    logging.info("Instantiating classifier")
    classifier = BERTTextSinglelabelClassifier(config, pdl1.cls_encoder, pdl1.tokenizer)

    # get the labeled data for train and dev partition
    # test partition already generated above
    train_ids, val_ids = train_test_split(
        all_labeled, test_size=0.25, random_state=config.seed
    )

    # get train data using train_ids
    # set of document SIDs was passed to split earlier
    train_df = all_data[all_data["CANDIDATE"].isin(train_ids)]
    train_df = condense_df(train_df, pdl1.cls_encoder, gold_types="test")

    val_df = all_data[all_data["CANDIDATE"].isin(val_ids)]
    val_df = condense_df(val_df, pdl1.cls_encoder, gold_types="test")

    # convert to dataset
    # for some reason here we need LABEL instead of LABELS
    train_dataset = Dataset.from_dict(
        {
            "texts": train_df["CANDIDATE"].tolist(),
            "label": train_df["GOLD"].tolist(),
        }
    )
    train_dataset = train_dataset.map(pdl1.tokenize, batched=True)

    val_dataset = Dataset.from_dict(
        {
            "texts": val_df["CANDIDATE"].tolist(),
            "label": val_df["GOLD"].tolist(),
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

    logging.info("TRAINING ON NER MODEL AND CLASSIFIER COMPLETE.")

    # optional step to run inference on test set at the end of training
    if config.evaluate_on_test_set:
        logging.info("evaluate_on_test_set is set to true. Running inference on test set.")
        # warning: this doesn't use VERY best model from CV for NER
        #   only best model from last fold of CV
        #   to run on very best model, use inference.py
        logging.info("RESULTS OF NER ON TEST PARTITION:")
        logging.info("BEST NER TRAINER ARGS:")
        logging.info(best_ner_trainer.args)
        logging.info("Now predicting on best ner trainer")
        test_results = best_ner_trainer.predict(test_dataset)

        # get only the items that were IDed as containing PD-L1 value
        logging.info("Now selecting only items with PDL1 value IDed by NER trainer")
        test_labeled_ids = id_labeled_items(
            test_results.predictions,
            test_dataset["texts"],
            pdl1.ner_encoder.transform(["O"]),
        )

        # subset the original dataframe with these SIDs
        logging.info("Subsetting original dataframe using only these IDed test items")
        test_df = all_data[all_data["CANDIDATE"].isin(test_labeled_ids)]
        test_df = condense_df(test_df, pdl1.cls_encoder, gold_types="test")

        new_test_dataset = Dataset.from_dict(
            {
                "texts": test_df["CANDIDATE"].tolist(),
                "label": test_df["GOLD"].tolist(),
            }
        )
        new_test_dataset = new_test_dataset.map(pdl1.tokenize, batched=True)

        logging.info("RESULTS OF CLASSIFICATION ON TEST DATA IDed IN STEP 1")
        cls_test_results = best_cls_trainer.predict(new_test_dataset)
        preds = torch.argmax(torch.Tensor(cls_test_results.predictions), dim=1)
        labels = new_test_dataset["label"]

        # get confusion matrix
        logging.info(f"Confusion matrix on test set: ")
        logging.info(confusion_matrix(labels, preds))

        # get classification report on val set
        logging.info(f"Classification report on test set: ")
        logging.info(
            classification_report(
                labels,
                preds,
                labels=np.array([i for i in range(len(pdl1.cls_encoder.classes_))]),
                target_names=pdl1.cls_encoder.classes_,
                zero_division=0.0,
            )
        )
