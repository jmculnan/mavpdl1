# train a multilabel classifier model

# IMPORT STATEMENTS
import evaluate
import numpy as np
import logging
import torch

from transformers import Trainer
from datasets import Dataset

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, classification_report

from config import train_config as config

# from other modules here
from mavpdl1.utils.utils import (
    get_tokenizer,
    tokenize_label_data,
    condense_df,
    CustomCallback,
)
from mavpdl1.preprocessing.data_preprocessing import PDL1Data
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

    # PART 2
    # USE THE IDENTIFIED PD-L1 INPUTS TO TRAIN A CLASSIFIER TO GET
    # VENDOR AND UNIT INFORMATION FROM THESE SAMPLES, WHEN AVAILABLE
    classifier = BERTTextMultilabelClassifier(config, label_enc_vendor_unit, tokenizer)

    # get the labeled data for train and dev partition
    # test partition already generated above
    train_ids, val_ids = train_test_split(
        ids_train_full, test_size=0.25, random_state=config.seed
    )

    # get train data using train_ids
    # set of document SIDs was passed to split earlier
    train_df = all_data[all_data["TIUDocumentSID"].isin(train_ids)]
    train_df = condense_df(train_df, label_enc_vendor_unit)

    val_df = all_data[all_data["TIUDocumentSID"].isin(val_ids)]
    val_df = condense_df(val_df, label_enc_vendor_unit)

    # convert to dataset
    # for some reason here we need LABEL instead of LABELS
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
    # add callback to print train compute_metrics for train set
    # in addition to val set
    classification_trainer.add_callback(CustomCallback(classification_trainer))

    # train the model
    classification_metrics = classification_trainer.train()

    logging.info("Results of our model on the validation dataset: ")
    # look at best model performance on validation dataset
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
    logging.info(classification_report(labels,
                                       preds,
                                       target_names=label_enc_vendor_unit.classes_,
                                       zero_division=0.0)
                 )

