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


class PDL1Data:
    def __init__(self, path_string):
        """
        :param path_string: The string path to the data csv
        """
        self.path = path_string
        self.data = self._read_in_data()

    def get_label_set(self, in_ner, in_classification=None, classification_type='multitask'):
        """
        Get the label sets for the PDL1 task
        These labels may be set up a couple of ways depending
        on task goal. The types are below:
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
            - in_ner = ['result'] *as of 24.02.08, bc unit is not
                given a separate annotation in the data
            - in_classification = ['test']
            - classification_type is ignored
        :param in_ner: list of class types to include
            options 'vendor', 'unit', 'result'. ALWAYS contains 'result'
        :param in_classification: optional list of class types
            options 'vendor', 'unit'
        :param classification_type: if in_classification isn't empty,
            whether to return a joined label set for multilabel
            or task-specific label sets for a multitask setup
        These two params together should have all three class types
        :return: two sets of labels, one for NER, one for classification
            if multitask classification, classification label set is a
            tuple of label sets for (vendor, unit)
        """
        # in_ner always has at least one item -- value
        ner_label_set = ["O"]
        for label_class in in_ner:
            ner_label_set.extend([f"B-{label_class}", f"I-{label_class}"])

        classification_label_set = []
        second_cls_label_set = None
        if in_classification:
            if 'vendor' in in_classification:
                classification_label_set.append("UNK_TEST")
                classification_label_set.extend(self.data["TEST"].dropna().unique().tolist())
            if 'unit' in in_classification:
                if classification_type == "multitask":
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
