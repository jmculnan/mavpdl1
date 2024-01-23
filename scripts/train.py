# train an NER model

# IMPORT STATEMENTS
import evaluate
import numpy as np

from transformers import Trainer, DataCollatorForTokenClassification
from datasets import Dataset

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold

from seqeval.scheme import IOB1
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

from config import train_config as config

# from other modules here
from mavpdl1.utils.utils import (
    get_tokenizer,
    tokenize_label_data,
    id_labeled_items,
    get_from_indexes,
)
from mavpdl1.preprocessing.data_preprocessing import PDL1Data
from mavpdl1.model.ner_model import BERTNER
from mavpdl1.model.classification_model import BERTTextClassifier

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
    label_set_vendor_unit = np.concatenate((all_data["TEST"].dropna().unique(),
                                            all_data["UNIT"].dropna().unique(),
                                            np.ndarray("None")))

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
    splits = config.num_splits
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
        data_collator = DataCollatorForTokenClassification(tokenizer)

        # set up trainer
        trainer = Trainer(
            model=ner.model,
            args=ner.training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            eval_dataset=val_dataset,
            compute_metrics=ner.compute_metrics,
        )

        # train the model
        metrics = trainer.train()

        # get the best predictions from the model
        # to select data inputs for classification task
        y_pred = trainer.predict(val_dataset)

        # IDENTIFY ALL SAMPLES THAT WILL MOVE ON TO BE USED AS INPUT WITH NEXT MODEL
        # define a second computation
        all_labeled_ids = id_labeled_items(
            y_pred.predictions,
            val_dataset["TIUDocumentSID"],
            label_enc_results.transform(["O"]),
        )
        all_labeled.extend(all_labeled_ids)

    print(all_labeled)

    # PART 2
    # USE THE IDENTIFIED PD-L1 INPUTS TO TRAIN A CLASSIFIER TO GET
    # VENDOR AND UNIT INFORMATION FROM THESE SAMPLES, WHEN AVAILABLE
    classifier = BERTTextClassifier(config, label_set_vendor_unit, tokenizer)

    # get the labeled data for train and dev partition
    # test partition already generated above
    train_ids, val_ids = train_test_split(ids_train_full, test_size=0.15, random_state=config.seed)

    # get train data using train_ids
    train_df = all_data[all_data["TIUDocumentSID"].isin(train_ids)]
    val_df = all_data[all_data["TIUDocumentSID"].isin(val_ids)]





    # SAVE RESULTS
    # ---------------------------------------------------------
    if config.save_plots:
        # save the training plots
        pass
