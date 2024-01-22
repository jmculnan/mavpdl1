# utilities functions for testing NER code
import torch
import uuid
import numpy as np

from transformers import BertTokenizerFast


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


def create_labels(text, tokens, start_idx, end_idx, labels, label):
    """
    Modified from Kyle code
    :param text: Full text string
    :param tokens: Tokenized text string
    :param start_idx: idx of char onset in text
    :param end_idx: idx of char offset in text
    :param labels:
    :param label:
    :return:
    """
    if labels is None:
        labels = label_encoder.transform(["O"] * 512)
    for i, offsets in enumerate(tokens["offset_mapping"][0]):
        if label and start_idx <= offsets[0] and offsets[1] <= end_idx:
            if label_encoder.inverse_transform([labels[i - 1]])[0] != "I-" + label:
                labels[i] = label_encoder.transform(["I-" + label])[0]
            else:
                labels[i] = label_encoder.transform(["B-" + label])[0]
    return text, labels


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
    all_sids = []  # todo: deidentified data doesn't contain this

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
                        for k in range(len(tok_ann)):
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
