# utilities functions for testing NER code
import torch
import uuid
import numpy as np
import pandas as pd
from copy import deepcopy

from transformers import BertTokenizerFast, TrainerCallback


def get_tokenizer(tokenizer_model="allenai/scibert_scivocab_uncased"):
    """
    Modified from Kyle code
    Can add params if multiple tokenizers allowed
    :return: A tokenizer with relevant tokens added
    """
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_model)
    # casing doesn't matter, added tokens can be cased or uncased
    # and tokenizer will make it uncased
    tokenizer.add_tokens(
        [
            "tps",
            "d22c3",
            "tc",
            "ic",
            "vsp263",
            "cps",
            "kp",
            "28-8",
            "id",
            "vsp142",
            "ta",
            "dako",
            "keytruda",
            "pembrolizumab",
            "pembro",
            "ventana",
            "sp142",
            "tecentriq",
            "atezolizumab",
            "imfinzi",
            "durvalumab",
            "sp263",
            "opdivo",
            "nivolumab",
        ]
    )

    return tokenizer


class CustomCallback(TrainerCallback):
    """
    To get predictions on train set at end of each epoch
    In order to print training curve + loss over epochs
    https://discuss.huggingface.co/t/metrics-for-training-set-in-trainer/2461/5
    """
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy


def tokenize_label_data(tokenizer, data_frame, label_encoder):
    """
    Use a pd df with columns ANNOTATION and CANDIDATE
    to get IOB-labeled, tokenized data
    :param tokenizer: a transformers tokenizer
    :param data_frame: pd df that has NOT been randomized
    :param label_encoder: a label encoder
    :return: tokenized input, IOB-formatted word-level ys, ids
    todo: label_encoder isn't really needed right now
        but should be more useful with a larger number
        of classes; think on if this is really the best
        place for it.
    """
    all_texts = []
    all_labels = []
    all_sids = []

    for i, row in data_frame.iterrows():
        # check if the row is a second annotation of the same example
        if not np.isnan(row["ANNOTATION_INDEX"]) and int(row["ANNOTATION_INDEX"]) > 0:
            tokenized = all_texts[-1]
            labels = all_labels[-1]
            del all_texts[-1]
            del all_labels[-1]
            del all_sids[-1]
        else:
            # tokenize the item
            tokenized = tokenizer.tokenize(row["CANDIDATE"])
            # generate O labels for max length
            labels = label_encoder.transform(["O"] * 512)
        # tokenize the annotation
        # this is set up with at most one annotation per row
        # if there is no annotation, just leave 'O' labels
        if type(row["ANNOTATION"]) == str:
            tok_ann = tokenizer.tokenize(row["ANNOTATION"])
            # TODO: find more accurate way to do this
            #   it's not guaranteed to always work
            # start with first occurrence
            for j in range(len(tokenized)):
                if tokenized[j : j + len(tok_ann)] == tok_ann:
                    labels[j] = label_encoder.transform(["B-result"])
                    if len(tok_ann) > 1:
                        for k in range(1, len(tok_ann)):
                            labels[j + k] = label_encoder.transform(["I-result"])
                    # we only take the first occurrence of this
                    # issue if we see a number twice but the
                    # actual label should be SECOND occurrence
                    break

        # add tokenized data + labels to full lists
        all_texts.append(row["CANDIDATE"])
        # convert to long to try to keep it from changing during training
        all_labels.append(labels)
        if "TIUDocumentSID" in data_frame.columns:
            all_sids.append(row["TIUDocumentSID"])
        else:
            # todo: remove after getting full data
            # to test code, use random uuid
            all_sids.append(str(uuid.uuid4()))

    return all_texts, all_labels, all_sids


def condense_df(df, label_encoder):
    """
    Condense a df where multiple rows have the same input text
    But different gold labels
    Concatenate the gold labels
    :param df: Data df
    :param label_encoder: LabelEncoder for TEST UNIT group
    :return:
    todo: if task formulation changes, we need this to be different
        as we'll have separate encoders for unit and test
    """
    # convert nan values to unk_test
    df["TEST"] = df["TEST"].apply(lambda x: "UNK_TEST" if pd.isnull(x) else x)
    # convert nan values to unk_unit
    df["UNIT"] = df["UNIT"].apply(lambda x: "UNK_UNIT" if pd.isnull(x) else x)

    listed = ['TEST', 'UNIT']
    condensed = df.groupby(['TIUDocumentSID', 'CANDIDATE'])[listed].agg(set).reset_index()

    condensed['GOLD'] = condensed.apply(lambda x: x['TEST'].union(x['UNIT']), axis=1)
    condensed['GOLD'] = condensed['GOLD'].apply(lambda x: vectorize_gold(x, label_encoder))

    return condensed


# vectorize gold labels
def vectorize_gold(gold_labels, label_encoder):
    """
    Vectorize gold labels for use in the multilabel classification model
    :param gold_labels: The set of gold labels for an item
    :param label_encoder: A label encoder
    :return:
    """
    # convert class str to int
    new_gold = label_encoder.transform(list(gold_labels)).tolist()
    # create vector of 0s
    gold_vec = [0] * len(label_encoder.classes_)

    # mentioned classes are 1, rest are 0
    for item in new_gold:
        gold_vec[item] = 1

    return gold_vec


def id_labeled_items(predictions, sids, o_class):
    """
    ID all texts that include at least one PDL1 value
    These texts will then be fed in for training of
    a second model that selects all those
    :param predictions: the set of model predictions
        predictions.shape() == (items, 512, classes)
    :param sids: the set of SIDs for the predictions
    :param o_class: the int number for the O class
    :return:
    # todo: add option to change acceptance threshold
    #   rather than just argmax
    """
    predicted_sids = []
    for i, item in enumerate(predictions):
        # get best prediction per item
        item = item.argmax(axis=1)
        # get SIDs with non-O predictions
        if set(list(item)) != set(o_class):
            predicted_sids.append(sids[i])

    return predicted_sids


def get_from_indexes(data, train_idxs, test_idxs):
    """
    Split data into train and test partitions using sets of indices
    :param data: The data
    :param train_idxs: Indices for train partition
    :param test_idxs: Indices for test partition
    :return:
    """
    train = []
    test = []
    for idx in train_idxs:
        train.append(data[idx])
    for idx in test_idxs:
        test.append(data[idx])

    return train, test
