# test the preprocessing steps for getting multilabel dataset
import evaluate
import numpy as np
import pandas as pd

from transformers import Trainer, DataCollatorForTokenClassification, DataCollatorWithPadding
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


if __name__ == "__main__":
    # use deidentified data sample
    data = PDL1Data("/Users/jculnan/va_data/pdl1_annotations-100_deidentified_fakeuuids.csv")
    all_data = data.data

    # # label set for vendor and unit
    label_set_vendor_unit = np.concatenate((all_data["TEST"].dropna().unique(),
                                            all_data["UNIT"].dropna().unique(),
                                            np.array(["Unk_test", "Unk_unit"])))

    print(label_set_vendor_unit)
    label_set_results = ["O", "B-result", "I-result"]

    label_enc_results = LabelEncoder().fit(label_set_results)
    label_enc_vendor_unit = LabelEncoder().fit(label_set_vendor_unit)

    # get tokenizer
    tokenizer = get_tokenizer("allenai/scibert_scivocab_uncased")

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
        tokenized, gold, sids, test_size=0.15, random_state=42
    )

    def tokize(text):
        return tokenizer(
            text["texts"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )

    # get the labeled data for train and dev partition
    # test partition already generated above
    train_ids, val_ids = train_test_split(ids_train_full, test_size=0.25, random_state=42)

    print(all_data.columns)
    # get train data using train_ids
    # set of document SIDs was passed to split earlier
    train_df = all_data[all_data["TIUDocumentSID"].isin(train_ids)]
    val_df = all_data[all_data["TIUDocumentSID"].isin(val_ids)]

    def condense_df(df):
        """
        Condense a df where multiple rows have the same input text
        But different gold labels
        Concatenate the gold labels
        :param df:
        :return:
        """
        # convert nan values to unk_test
        df["TEST"] = df["TEST"].apply(lambda x: "UNK_TEST" if pd.isnull(x) else x)
        # convert nan values to unk_unit
        df["UNIT"] = df["UNIT"].apply(lambda x: "UNK_UNIT" if pd.isnull(x) else x)

        listed = ['TEST', 'UNIT']
        return df.groupby(['TIUDocumentSID', 'CANDIDATE'])[listed].agg(set)

    print(condense_df(train_df))
    exit()

    # vectorize gold labels

    def vectorize_gold(gold_labels, label_encoder):
        """
        Vectorize gold labels for use in the multilabel classification model
        :param gold_labels: The
        :param label_encoder:
        :return:
        """

    #
    # train_dataset = Dataset.from_dict(
    #     {"texts": , "labels": y_train, "TIUDocumentSID": }
    # )
    # train_dataset = train_dataset.map(tokize, batched=True)
    #
    # val_dataset = Dataset.from_dict(
    #     {"texts": X_val, "labels": y_val, "TIUDocumentSID": ids_val}
    # )
    # val_dataset = val_dataset.map(tokize, batched=True)
    #
    # # convert to dataset
    #
    # # set training args