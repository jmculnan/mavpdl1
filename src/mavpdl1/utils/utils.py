# utilities functions for testing NER code
import pandas as pd
from copy import deepcopy
import logging

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

    def on_train_begin(self, args, state, control, **kwargs):
        logging.info("Starting to train a new model. Hyperparameters are: ")
        logging.info(state.trial_params)

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            train_set_results = self._trainer.evaluate(
                eval_dataset=self._trainer.train_dataset, metric_key_prefix="train"
            )
            dev_set_results = self._trainer.evaluate(
                eval_dataset=self._trainer.eval_dataset, metric_key_prefix="eval"
            )

            logging.info(train_set_results)
            logging.info(dev_set_results)

            return control_copy


class CustomCallbackNER(TrainerCallback):
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
            train_set_results = self._trainer.evaluate(
                eval_dataset=self._trainer.train_dataset, metric_key_prefix="train"
            )
            dev_set_results = self._trainer.evaluate(
                eval_dataset=self._trainer.eval_dataset, metric_key_prefix="eval"
            )

            logging.info(train_set_results)
            logging.info(dev_set_results)

            return control_copy


def optuna_hp_space(trial):
    """
    Get a hyperparameter space for optuna backend to use with a
    trainer trial
    ONLY used with classifier currently
    :param trial: a trainer trial
    :return:
    """
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [1, 2, 4]),
        "num_train_epochs": trial.suggest_categorical("num_train_epochs", [1, 2, 4])
    }


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


def condense_df(df, label_encoder, gold_types="both"):
    """
    Condense a df where multiple rows have the same input text
    But different gold labels
    Concatenate the gold labels
    :param df: Data df
    :param label_encoder: LabelEncoder for TEST UNIT group
    :param gold_types: list of the types of gold labels we need
        this may be 'test', 'unit' or 'both'
    :return:
    todo: if task formulation changes, we need this to be different
        as we'll have separate encoders for unit and test
    """
    # convert nan values to unk_test
    df["TEST"] = df["TEST"].apply(lambda x: "UNK_TEST" if pd.isnull(x) else x)
    # convert nan values to unk_unit
    df["UNIT"] = df["UNIT"].apply(lambda x: "UNK_UNIT" if pd.isnull(x) else x)

    listed = ["TEST", "UNIT"]
    condensed = (
        df.groupby(["CANDIDATE"])[listed].agg(set).reset_index()
    )

    if gold_types == "both":
        condensed["GOLD"] = condensed.apply(
            lambda x: x["TEST"].union(x["UNIT"]), axis=1
        )
        condensed["GOLD"] = condensed["GOLD"].apply(
            lambda x: vectorize_gold(x, label_encoder)
        )
    if gold_types == "test":
        # todo: see if there are ever instances of
        #     multiple tests in a single input
        condensed["GOLD"] = condensed["TEST"]
        condensed["GOLD"] = condensed["GOLD"].apply(
            lambda x: transform_gold(x, label_encoder)
        )
    elif gold_types == "unit":
        # there are frequently multiple UNITS in an input
        # so you MUST treat this as multilabel
        condensed["GOLD"] = condensed["UNIT"]
        condensed["GOLD"] = condensed["GOLD"].apply(
            lambda x: vectorize_gold(x, label_encoder)
        )
    else:
        logging.error(f"Gold type {str(gold_types)} unknown.")

    return condensed


def transform_gold(gold_label, label_encoder):
    """
    Use a label encoder to transform text label to gold
    :param gold_label:
    :param label_encoder:
    :return:
    """
    if len(gold_label) > 1:
        logging.error(
            "MULTIPLE GOLD LABELS FOUND, BUT THIS IS BEING TREATED AS SINGLE-LABEL TASK"
        )
        exit("Verify that you are using 'test'")
    else:
        gold = list(gold_label)[0]
    label = label_encoder.transform([gold])
    return int(label)


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
