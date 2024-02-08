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
import uuid

from sklearn.preprocessing import LabelEncoder

from mavpdl1.utils.utils import get_tokenizer


class PDL1Data:
    def __init__(self, path_string, model, ner_classes, classification_classes=None, classification_type='multitask'):
        """
        An object used to load and prepare data for input into a network
        :param path_string: The string path to the data csv
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
        1. Have NER for vendor, unit, and value separate
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
        self.path = path_string
        self.data = self._read_in_data()

        self.in_ner = ner_classes
        self.in_classification = classification_classes
        self.classification_type = classification_type

        # get set of NER and classification labels
        self.ner_labels, self.cls_labels = self._get_label_set()

        # get tokenizer
        self.tokenizer = get_tokenizer(model)

        # get label encoders
        self.ner_encoder = LabelEncoder().fit(self.ner_labels)
        self.cls_encoder = LabelEncoder().fit(self.cls_labels[0] if type(self.cls_labels) == tuple else self.cls_labels)
        self.cls_encoder2 = LabelEncoder().fit(self.cls_labels[1]) if type(self.cls_labels) == tuple else None

    def _get_label_set(self):
        """
        Get the label sets for the PDL1 task
        :return: two sets of labels, one for NER, one for classification
            if multitask classification, classification label set is a
            tuple of label sets for (vendor, unit)
        """
        # in_ner always has at least one item -- value
        ner_label_set = ["O"]
        if 'result' in self.in_ner:
            ner_label_set.extend(["B-result", "I-result"])
        if 'unit' in self.in_ner:
            ner_label_set.extend(self.data["UNIT"].dropna().unique().tolist())

        classification_label_set = []
        second_cls_label_set = None
        if self.in_classification:
            if 'vendor' in self.in_classification:
                classification_label_set.append("UNK_TEST")
                classification_label_set.extend(self.data["TEST"].dropna().unique().tolist())
            if 'unit' in self.in_classification:
                if self.classification_type == "multitask":
                    second_cls_label_set = ["UNK_UNIT"]
                    second_cls_label_set.extend(self.data["UNIT"].dropna().unique().tolist())
                else:
                    classification_label_set.append("UNK_UNIT")
                    classification_label_set.extend(self.data["UNIT"].dropna().unique().tolist())

        if second_cls_label_set:
            return np.array(ner_label_set), \
                (np.array(classification_label_set), np.array(second_cls_label_set))
        else:
            return np.array(ner_label_set), \
                np.array(classification_label_set)

    def tokenize_label_data(self, tokenizer, data_frame, label_encoder):
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
                del all_texts[-1], all_labels[-1], all_sids[-1]
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
                    if tokenized[j: j + len(tok_ann)] == tok_ann:
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

    def _read_in_data(self):
        """
        Read in the data
        :return:
        """
        data = pd.read_csv(self.path)

        # todo: this depends on the type of task setup we have for classification
        # if we use multilabel, then this is appropriate
        # if not,
        data[["TEST", "UNIT"]] = data.apply(
            lambda x: convert_label(x["LABELS"]), axis=1, result_type="expand"
        )

        # get subset of data
        # we only need start, end, annotation_index,
        # test, unit, annotation, and candidate
        try:
            data = data[
                [
                    "START",
                    "END",
                    "ANNOTATION_INDEX",
                    "ANNOTATION",
                    "TEST",
                    "UNIT",
                    "CANDIDATE",
                    "TIUDocumentSID",
                ]
            ]
        # if the TIUDocumentSID is not present
        except ValueError:
            data = data[
                [
                    "START",
                    "END",
                    "ANNOTATION_INDEX",
                    "ANNOTATION",
                    "TEST",
                    "UNIT",
                    "CANDIDATE",
                ]
            ]

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
