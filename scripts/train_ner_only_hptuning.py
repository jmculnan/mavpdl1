# train an NER model

# IMPORT STATEMENTS
import evaluate
import pandas as pd
import torch
import logging
import random

from transformers import Trainer, DataCollatorForTokenClassification
from datasets import Dataset

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix

from seqeval.metrics import classification_report

from config import train_config as config

# from other modules here
from mavpdl1.utils.utils import (
    get_from_indexes,
    CustomCallback,
    id_labeled_items,
)

from mavpdl1.preprocessing.data_preprocessing import PDL1Data
from mavpdl1.model.ner_model import BERTNER

# load seqeval in evaluate
seqeval = evaluate.load("seqeval")


def calc_best_hyperparams(list_of_param_dicts):
    """
    Using the best hyperparameters of each inner loop,
    as determined by outer loop holdout performance,
    find best parameters through averaging
    :param list_of_param_dicts:
    :return:
    """
    # names of hyperparameters that must be ints
    # todo: add others as needed
    logging.info("There are currently only two hyperparameters listed"
                 "as needing to be ints. If you include a hyperparameter"
                 "in the search that needs to be an int and is NOT "
                 "number of train epochs or per device train batch size, "
                 "please add this to 'int_params' in function"
                 "calc_best_hyperparams")
    int_params = ["num_train_epochs", "per_device_train_batch_size"]
    # holder for best parameters
    best_params = {}
    for param_dict in list_of_param_dicts:
        for param, p_val in param_dict.items():
            if param not in best_params:
                best_params[param] = [p_val]
            else:
                best_params[param].append(p_val)
    for param in best_params.keys():
        if param in int_params:
            p_val = round(sum(best_params[param]) / float(len(best_params[param])))
        else:
            p_val = sum(best_params[param]) / float(len(best_params[param]))
        best_params[param] = p_val

    return best_params


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
    tokenized, gold= pdl1.tokenize_label_data()

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
    test_dataset = Dataset.from_dict({"texts": X_test, "label": y_test})
    test_dataset = test_dataset.map(pdl1.tokenize, batched=True)

    # # generate train and dev for initial hyperparameter search
    # (
    #     X_train,
    #     X_val,
    #     y_train,
    #     y_val) = train_test_split(
    #     X_train_full, y_train_full, test_size=0.2, random_state=config.seed
    # )

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

        # set validation dataset for the outer loop
        val_dataset_outer = Dataset.from_dict(
            {"texts": X_val, "labels": y_val}
        )
        val_dataset_outer = val_dataset_outer.map(pdl1.tokenize, batched=True)

        best_f1 = -1.0

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

            # TRAIN THE MODEL
            # ---------------------------------------------------------
            # we need this to be a 2 part model
            # part 1: IOB NER task over the data to ID results; this will need to be trained as a first step
            # part 2: AFTER training the NER system, we need to select only those items IDed by the NER
            #   and use those in the classification task
            ner.update_save_path(f"{config.savepath}/ner/fold_{i}")
            ner.update_log_path(f"{config.ner_logging_dir}/fold_{i}")

            # reinit the model
            ner.reinit_model()

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
                logging.info("HERE ARE THE USED HYPERPARAMETERS")
                logging.info(used_hyperparameters)

            else:
                print(used_hyperparameters)
                # get hyperparameters from used_hyperparameters
                new_hyperparams = used_hyperparameters[j]

            logging.info(f"Hyperparameters for fold {i}, inner fold {j}:")
            logging.info(new_hyperparams)

            # add new hyperparameters to model training args
            ner.load_new_hyperparameters(new_hyperparams)

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
            trainer.add_callback(CustomCallback(trainer))

            logging.info("Beginning training loop")
            inner_trained = trainer.train()

            # get predictions on val dataset from outer loop
            y_pred = trainer.predict(val_dataset_outer)
            labels = val_dataset_outer["labels"]
            metrics = ner.compute_metrics([y_pred.predictions, labels])

            logging.info("Results of inner loop hyperparameters on outer loop holdout")
            logging.info(metrics)

            # record these hyperparameters if they improve performance
            if metrics["f1"] > best_f1:
                ner.save_best_hyperparameters(new_hyperparams)
                best_f1 = metrics["f1"]

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
        all_labeled_ids = id_labeled_items(
            y_pred.predictions,
            val_dataset["texts"],
            pdl1.ner_encoder.transform(["O"]),
        )
        all_labeled.extend(all_labeled_ids)

        predictions = torch.argmax(torch.from_numpy(y_pred.predictions), dim=2)
        labels = [list(map(int, label)) for label in val_dataset["labels"]]

        true_labels = [pdl1.ner_encoder.inverse_transform(label) for label in labels]
        true_predictions = [
            pdl1.ner_encoder.inverse_transform(prediction) for prediction in predictions
        ]

        true_predictions = list(map(list, true_predictions))
        true_labels = list(map(list, true_labels))

        preds_for_confusion = predictions.flatten().squeeze().tolist()
        labels_for_confusion = [label for label_list in labels for label in label_list]

        logging.info(f"Confusion matrix on validation set: ")
        logging.info(confusion_matrix(labels_for_confusion, preds_for_confusion))

        report = classification_report(true_labels, true_predictions, digits=2)
        logging.info(pdl1.ner_encoder.classes_)

        logging.info(report)

    # save the list of all documents containing PD-L1 values according to the model
    labeled_df = pd.DataFrame(all_labeled, columns=["CANDIDATE"])
    labeled_df.to_csv(
        f"{config.savepath}/all_documents_with_IDed_pdl1_values.csv", index=False
    )
