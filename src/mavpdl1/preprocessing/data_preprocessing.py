# prepare data for use in network (from deidentified csv)
# data has the following columns
# unnamed index (can remove)
# LABELS -- unit and vendor labels;
#   formats VENDOR_UNIT (both)
#           VENDOR_UU   (vendor only)
#           UNIT_UT     (unit only)
#           UT_UU       (no unit or vendor)
# START -- start of PD-L1 value from text (in # chars)
# END -- end of PD-L1 value from text (in # chars)
# ANNOTATION_INDEX -- 0 if first annotation in a text, 1 if second
# ANNOTATION -- PD-L1 value
# CANDIDATE -- full extracted text
# INAVLID_CHARS -- BOOL (only true once in sample)
# MULTIPLE LABELS -- BOOL (always false)
# OVERLAPS -- BOOL (alwyas false or missing)

# import statements
import pandas as pd
import numpy as np
import logging

from sklearn.preprocessing import LabelEncoder

from mavpdl1.utils.utils import get_tokenizer


class PDL1Data:
    def __init__(
        self,
        path_strings,
        model,
        ner_classes,
        classification_classes=None,
        classification_type="multitask",
    ):
        """
        An object used to load and prepare data for input into a network
        :param path_strings: List of string paths to the data csvs
        :param model: the model name (used to instantiate tokenizer)
        :param ner_classes: list of class types to include
            options 'vendor', 'unit', 'result'
        :param classification_classes: optional list of class types
            options 'vendor', 'unit'
        :param classification_type: if classification_classes isn't empty,
            whether to return a joined label set for multilabel
            or task-specific label sets for a multitask setup
        ner_classes, classification_classes, and classification_type may
        have several options depending on task goal. Current task set-ups
        are as follows:
        1. Have NER for vendor, unit, and value separately
            TODO: this is NOT implemented, as we do not have gold annotations
                for each of these types. Dataframe format may change once
                these are added.
            - in_ner = ['test', 'unit', 'result']
            - in_classification = None
            - classification_type is ignored
        2. Have NER for value, classification for vendor and unit
            a. multilabel classification
            - in_ner = ['result']
            - in_classification = ['test', 'unit']
            - classification_type = 'multilabel'
            b. single-label multitask classification
            - in_ner = ['result']
            - in_classification = ['test', 'unit']
            - classification_type = 'multitask'
        3. Have NER for value + unit, classification for vendor
            - in_ner = ['unit'] *as of 24.02.08, bc unit is not
                given a separate annotation from value in the data
            - in_classification = ['test']
            - classification_type is ignored
        """
        self.path = path_strings
        self.data = self._read_in_data()

        self.in_ner = ner_classes
        self.in_classification = classification_classes
        self.classification_type = classification_type

        # get set of NER and classification labels
        self.ner_labels, self.cls_labels = self._get_label_set()
        logging.info("Label set created--labels are: ")
        logging.info("NER Labels: ")
        logging.info(self.ner_labels.tolist())
        logging.info("CLS Labels: ")
        logging.info(self.cls_labels.tolist())

        # get tokenizer
        self.tokenizer = get_tokenizer(model)
        logging.info(f"Tokenizer loaded from {model}")

        # get label encoders
        self.ner_encoder = LabelEncoder().fit(self.ner_labels)
        self.cls_encoder = LabelEncoder().fit(
            self.cls_labels[0] if type(self.cls_labels) == tuple else self.cls_labels
        )
        self.cls_encoder2 = (
            LabelEncoder().fit(self.cls_labels[1])
            if type(self.cls_labels) == tuple
            else None
        )
        logging.info("Label encoders fit")

    def _get_label_set(self):
        """
        Get the label sets for the PDL1 task
        :return: two sets of labels, one for NER, one for classification
            if multitask classification, classification label set is a
            tuple of label sets for (vendor, unit)
        """
        # in_ner always has exactly one item -- either result or unit
        ner_label_set = ["O"]
        if "result" in self.in_ner:
            ner_label_set.extend(["B-result", "I-result"])
        elif "unit" in self.in_ner:
            units = self.data["UNIT"].dropna().unique().tolist()
            unitset = [[f"B-{item}", f"I-{item}"] for item in units]
            ner_label_set.extend([item for units in unitset for item in units])
        else:
            logging.error(f"Cannot get label set for set {self.in_ner}.")

        classification_label_set = []
        second_cls_label_set = None
        if self.in_classification:
            if "test" in self.in_classification:
                classification_label_set.append("UNK_TEST")
                classification_label_set.extend(
                    self.data["TEST"].dropna().unique().tolist()
                )
            if "unit" in self.in_classification:
                if self.classification_type == "multitask":
                    second_cls_label_set = ["UNK_UNIT"]
                    second_cls_label_set.extend(
                        self.data["UNIT"].dropna().unique().tolist()
                    )
                else:
                    classification_label_set.append("UNK_UNIT")
                    classification_label_set.extend(
                        self.data["UNIT"].dropna().unique().tolist()
                    )

        if second_cls_label_set:
            return np.array(ner_label_set), (
                np.array(classification_label_set),
                np.array(second_cls_label_set),
            )
        else:
            return np.array(ner_label_set), np.array(classification_label_set)

    def tokenize(self, texts):
        # tokenizes the texts in a Dataset
        return self.tokenizer(
            texts["texts"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )

    def tokenize_label_data(self):
        """
        Tokenizes and labels the data in self.data
        to get IOB-labeled, tokenized data
        :return: tokenized input, IOB-formatted word-level ys, ids
        todo: add in functionality to accept multitask NER
        """
        logging.info("Beginning data tokenization and NER label preparation")

        all_texts = []
        all_labels = []

        for i, row in self.data.iterrows():
            # check if the row is a second annotation of the same example
            if (
                not np.isnan(row["ANNOTATION_INDEX"])
                and int(row["ANNOTATION_INDEX"]) > 0
            ):
                tokenized = all_texts[-1]
                labels = all_labels[-1]
                del all_texts[-1], all_labels[-1]
            else:
                # tokenize the item
                tokenized = self.tokenizer.tokenize(row["CANDIDATE"])
                # generate O labels for max length
                labels = self.ner_encoder.transform(["O"] * 512)
            # tokenize the annotation
            # this is set up with at most one annotation per row
            # if there is no annotation, just leave 'O' labels
            if type(row["ANNOTATION"]) == str:
                tok_ann = self.tokenizer.tokenize(row["ANNOTATION"])
                # TODO: find more accurate way to do this
                #   it's not guaranteed to always work
                # start with first occurrence
                for j in range(len(tokenized)):
                    if tokenized[j : j + len(tok_ann)] == tok_ann:
                        # if we are only labeling 'result', we don't look at other
                        # columns in the data, just transform to B-result
                        if len(self.in_ner) == 1:
                            if self.in_ner[0] == "result":
                                labels[j] = self.ner_encoder.transform(["B-result"])
                            # else, we need to use the appropriate column from data
                            # todo: this assumes AT MOST one task is in the NER
                            #   out of test, unit; will we ever need BOTH?
                            elif (
                                self.in_ner[0] == "unit" and type(row["UNIT"]) != float
                            ):
                                labels[j] = self.ner_encoder.transform(
                                    [f"B-{row['UNIT']}"]
                                )
                            elif (
                                self.in_ner[0] == "test" and type(row["TEST"]) != float
                            ):
                                labels[j] = self.ner_encoder.transform(
                                    [f"B-{row['TEST']}"]
                                )
                        else:
                            logging.error(
                                "Multiple items in self.in_ner--currently only use one NER task at a time."
                            )

                    if len(tok_ann) > 1:
                        for k in range(1, len(tok_ann)):
                            if len(self.in_ner) == 1:
                                if self.in_ner[0] == "result":
                                    labels[j + k] = self.ner_encoder.transform(
                                        ["I-result"]
                                    )
                                elif (
                                    self.in_ner[0] == "unit"
                                    and type(row["UNIT"]) != float
                                ):
                                    labels[j] = self.ner_encoder.transform(
                                        [f"I-{row['UNIT']}"]
                                    )
                                elif (
                                    self.in_ner[0] == "test"
                                    and type(row["TEST"]) != float
                                ):
                                    labels[j] = self.ner_encoder.transform(
                                        [f"I-{row['TEST']}"]
                                    )
                            else:
                                logging.error(
                                    "Multiple items in self.in_ner--currently only use one NER task at a time."
                                )

                    # we only take the first occurrence of this
                    # issue if we see a number twice but the
                    # actual label should be SECOND occurrence
                    break

            # add tokenized data + labels to full lists
            all_texts.append(row["CANDIDATE"])

            # convert to long to try to keep it from changing during training
            all_labels.append(labels)

        logging.info("Tokenization and NER label preparation complete")

        return all_texts, all_labels

    def _read_in_data(self):
        """
        Read in the data
        :return:
        """
        data = None
        logging.info("Reading in data")
        if len(self.path) == 0:
            logging.error("No data to read in")
        elif type(self.path) != list:
            logging.error("Data is not provided as a list!")
        for strpath in self.path:
            if type(strpath) != str:
                logging.error(f"Path {str(strpath)} is not a string; this data cannot be read in")
            if data is None:
                data = pd.read_csv(strpath)
            else:
                data = pd.concat([data, pd.read_csv(strpath)])

        logging.info("Finished reading data.")

        # separate test_unit labels into test and unit columns
        TEST_UNIT = data["LABELS"].apply(lambda x: convert_label(x) if x else (None, None))
        TEST = [x[0] if x else None for x in TEST_UNIT]
        UNIT = [x[1] if x else None for x in TEST_UNIT]
        data["TEST"] = TEST
        data["UNIT"] = UNIT
        logging.info("TEST and UNIT columns created from LABELS column")

        # get subset of data
        # we only need start, end, annotation_index,
        # test, unit, annotation, and candidate
        data = data[
            [
                "START",
                "END",
                "ANNOTATION_INDEX",
                "ANNOTATION",
                "TEST",
                "UNIT",
                "CANDIDATE",
                "LABELS",
            ]
        ]

        logging.info("Data read-in and TEST, UNIT column creation complete")

        return data


def convert_label(label_string):
    """
    Convert LABELS column to TEST and UNIT columns
    :param label_string: A single TEST_UNIT label string
    :return: updated dataframe
    """
    # if labels is nan, test and unit are nan also
    # if vendor and unit are unknown, also nan
    if type(label_string) != str or label_string == "UT_UU":
        return [np.nan, np.nan]
    # if unit is unknown, make it nan
    elif label_string.endswith("UU"):
        return [label_string.split("_")[0], np.nan]
    # if vendor is unknown, make it nan
    # order swapped if vendor is unknown but unit is found
    elif label_string != "UT_UU" and "UT" in label_string:
        return [np.nan, label_string.split("_")[0]]
    # else return both tags
    else:
        return label_string.split("_")
