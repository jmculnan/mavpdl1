# train an NER model

# IMPORT STATEMENTS
import evaluate
import numpy as np
import logging
import torch

from transformers import (
    Trainer,
    DataCollatorForTokenClassification,
)
from datasets import Dataset

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import multilabel_confusion_matrix, classification_report

from config import train_config as config

# from other modules here
from mavpdl1.utils.utils import (
    get_tokenizer,
    tokenize_label_data,
    id_labeled_items,
    get_from_indexes,
    condense_df,
    CustomCallback,
)
from mavpdl1.preprocessing.data_preprocessing import PDL1Data
from mavpdl1.model.ner_model import BERTNER
from mavpdl1.model.classification_model import BERTTextMultilabelClassifier

# load seqeval in evaluate
seqeval = evaluate.load("seqeval")


if __name__ == "__main__":
    # PREPARE DATA
    # ---------------------------------------------------------
    # use deidentified data sample
    data = PDL1Data(config.dataset_location)
    all_data = data.data

    # get label sets
    # label set for test results is just O, B-result, I-result
    label_set_results = ["O", "B-result", "I-result"]
    # # label set for vendor and unit
    label_set_vendor_unit = np.concatenate(
        (
            all_data["TEST"].dropna().unique(),
            all_data["UNIT"].dropna().unique(),
            np.array(["UNK_TEST", "UNK_UNIT"]),
        )
    )

    # encoders for the labels
    label_enc_results = LabelEncoder().fit(label_set_results)
    label_enc_vendor_unit = LabelEncoder().fit(label_set_vendor_unit)

    # get tokenizer
    tokenizer = get_tokenizer(config.model)

    # get tokenized data and IOB-2 gold labeled data
    tokenized, gold, sids = tokenize_label_data(tokenizer, all_data, label_enc_results)

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

    # tokenize function for dataset mapping
    def tokize(text):
        return tokenizer(
            text["texts"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )

    # generate the test dataset if needed
    # train and val done below due to nature of the tasks
    test_dataset = Dataset.from_dict({"texts": X_test, "TIUDocumentSID": ids_test})
    test_dataset = test_dataset.map(tokize, batched=True)

    # convert train data into KFold splits
    splits = config.ner_num_splits
    folds = KFold(n_splits=splits, random_state=config.seed, shuffle=True)
    idxs = range(len(X_train_full))

    # create a holder for all items that you've IDed as
    # having a PD-L1 value
    all_labeled = []

    # do CV over the data
    for i, (train_index, test_index) in enumerate(folds.split(idxs)):
        X_train, X_val = get_from_indexes(X_train_full, train_index, test_index)
        y_train, y_val = get_from_indexes(y_train_full, train_index, test_index)
        ids_train, ids_val = get_from_indexes(ids_train_full, train_index, test_index)

        # todo: kyle code used `label` but on CPU would not train
        #   unless i changed it to `labels`
        train_dataset = Dataset.from_dict(
            {"texts": X_train, "labels": y_train, "TIUDocumentSID": ids_train}
        )
        train_dataset = train_dataset.map(tokize, batched=True)

        val_dataset = Dataset.from_dict(
            {"texts": X_val, "labels": y_val, "TIUDocumentSID": ids_val}
        )
        val_dataset = val_dataset.map(tokize, batched=True)

        # TRAIN THE MODEL
        # ---------------------------------------------------------
        # we need this to be a 2 part model
        # part 1: IOB NER task over the data to ID results; this will need to be trained as a first step
        # part 2: AFTER training the NER system, we need to select only those items IDed by the NER
        #   and use those in the classification task

        # PART 1
        ner = BERTNER(config, label_enc_results, tokenizer)

        # add data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer, padding=True, return_tensors="pt"
        )

        ner.update_save_path(f"{config.savepath}/ner/fold_{i}")
        ner.update_log_path(f"{config.ner_logging_dir}/fold_{i}")

        # set up trainer
        trainer = Trainer(
            model=ner.model,
            args=ner.training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            eval_dataset=val_dataset,
            compute_metrics=ner.compute_metrics,
        )
        # add callback to print train compute_metrics for train set
        # in addition to val set
        trainer.add_callback(CustomCallback(trainer))

        # train the model
        metrics = trainer.train()

        # get the best predictions from the model
        # to select data inputs for classification task
        print("BEST NER PREDICTIONS ON VALIDATION SET: ")
        y_pred = trainer.predict(val_dataset)

        # IDENTIFY ALL SAMPLES THAT WILL MOVE ON TO BE USED AS INPUT WITH NEXT MODEL
        # define a second computation
        all_labeled_ids = id_labeled_items(
            y_pred.predictions,
            val_dataset["TIUDocumentSID"],
            label_enc_results.transform(["O"]),
        )
        all_labeled.extend(all_labeled_ids)

    # print(all_labeled)
    logging.info("CV WITH NER MODEL COMPLETE. NOW BEGINNING CLASSIFICATION TASK")
    # PART 2
    # USE THE IDENTIFIED PD-L1 INPUTS TO TRAIN A CLASSIFIER TO GET
    # VENDOR AND UNIT INFORMATION FROM THESE SAMPLES, WHEN AVAILABLE
    classifier = BERTTextMultilabelClassifier(config, label_enc_vendor_unit, tokenizer)

    # get the labeled data for train and dev partition
    # test partition already generated above
    train_ids, val_ids = train_test_split(
        all_labeled, test_size=0.25, random_state=config.seed
    )

    # get train data using train_ids
    # set of document SIDs was passed to split earlier
    train_df = all_data[all_data["TIUDocumentSID"].isin(train_ids)]
    train_df = condense_df(train_df, label_enc_vendor_unit)

    val_df = all_data[all_data["TIUDocumentSID"].isin(val_ids)]
    val_df = condense_df(val_df, label_enc_vendor_unit)

    # convert to dataset
    # todo: here, we need label instead of labels to run!
    train_dataset = Dataset.from_dict(
        {
            "texts": train_df["CANDIDATE"].tolist(),
            "label": train_df["GOLD"].tolist(),
            "TIUDocumentSID": train_df["TIUDocumentSID"].tolist(),
        }
    )
    train_dataset = train_dataset.map(tokize, batched=True)

    val_dataset = Dataset.from_dict(
        {
            "texts": val_df["CANDIDATE"].tolist(),
            "label": val_df["GOLD"].tolist(),
            "TIUDocumentSID": val_df["TIUDocumentSID"].tolist(),
        }
    )
    val_dataset = val_dataset.map(tokize, batched=True)

    # instantiate trainer
    classification_trainer = Trainer(
        model=classifier.model,
        args=classifier.training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=classifier.multilabel_compute_metrics,
    )

    # train the model
    classification_metrics = classification_trainer.train()

    print("BEST CLASSIFIER PREDICTIONS ON VALIDATION SET: ")
    y_pred = classification_trainer.predict(val_dataset)

    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(y_pred.predictions))
    # next, use threshold to turn them into integer predictions
    preds = np.zeros(probs.shape)
    # use 0.5 as a threshold for predictions
    preds[np.where(probs >= 0.5)] = 1

    labels = val_dataset["label"]
    labels = [list(map(int, label)) for label in labels]

    # get confusion matrix
    logging.info(f"Confusion matrix on validation set: ")
    logging.info(multilabel_confusion_matrix(labels, preds))

    # get classification report on val set
    logging.info(f"Classification report on validation set: ")
    logging.info(
        classification_report(
            labels,
            preds,
            target_names=label_enc_vendor_unit.classes_,
            zero_division=0.0,
        )
    )

    logging.info("TRAINING ON NER MODEL AND CLASSIFIER COMPLETE.")

    # optional step to run inference on test set at the end of training
    if config.evaluate_on_test_set:
        # warning: this doesn't use VERY best model from CV for NER
        #   only best model from last fold of CV
        #   to run on very best model, use inference.py
        print("RESULTS OF NER ON TEST PARTITION:")
        test_results = trainer.predict(test_dataset)

        # get only the items that were IDed as containing PD-L1 value
        test_labeled_ids = id_labeled_items(
            test_results.predictions,
            test_dataset["TIUDocumentSID"],
            label_enc_results.transform(["O"]),
        )

        # subset the original dataframe with these SIDs
        test_df = all_data[all_data["TIUDocumentSID"].isin(test_labeled_ids)]
        test_df = condense_df(test_df, label_enc_vendor_unit)

        new_test_dataset = Dataset.from_dict(
            {
                "texts": test_df["CANDIDATE"].tolist(),
                "label": test_df["GOLD"].tolist(),
                "TIUDocumentSID": test_df["TIUDocumentSID"].tolist(),
            }
        )
        new_test_dataset = new_test_dataset.map(tokize, batched=True)

        print("RESULTS OF CLASSIFICATION ON TEST DATA IDed IN STEP 1")
        cls_test_results = classification_trainer.predict(new_test_dataset)

        # todo: turn this into a fuction
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(cls_test_results.predictions))
        # next, use threshold to turn them into integer predictions
        preds = np.zeros(probs.shape)
        # use 0.5 as a threshold for predictions
        preds[np.where(probs >= 0.5)] = 1

        labels = new_test_dataset["label"]
        labels = [list(map(int, label)) for label in labels]

        # get confusion matrix
        logging.info(f"Confusion matrix on test set: ")
        logging.info(multilabel_confusion_matrix(labels, preds))

        # get classification report on val set
        logging.info(f"Classification report on test set: ")
        logging.info(
            classification_report(
                labels,
                preds,
                target_names=label_enc_vendor_unit.classes_,
                zero_division=0.0,
            )
        )
