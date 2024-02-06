# perform inference on the ner + classifier models
# import statements
import pandas as pd
import torch
import json
import evaluate
import logging
import numpy as np

from transformers import (
    BertForTokenClassification,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from seqeval.scheme import IOB1
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

from config import train_config as config

from mavpdl1.preprocessing.data_preprocessing import PDL1Data
from mavpdl1.model.ner_model import BERTNER
from mavpdl1.model.classification_model import BERTTextMultilabelClassifier

# from other modules here
from mavpdl1.utils.utils import (
    get_tokenizer,
    tokenize_label_data,
    get_from_indexes,
    CustomCallback,
)


if __name__ == "__main__":

    # read in data
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

    # get tokenizer
    tokenizer = get_tokenizer(config.model)

    # get tokenized data and IOB-2 gold labeled data
    tokenized, gold, sids = tokenize_label_data(tokenizer, all_data, label_enc_results)

    # convert data to train, dev, and test
    # using same percentage and config as in training script
    # to ensure we end up with the same data splits
    _, X_test, _, y_test, _, ids_test = train_test_split(
        tokenized, gold, sids, test_size=0.15, random_state=config.seed
    )

    # either read in trained model or train the model


    # generate predictions on test partition
