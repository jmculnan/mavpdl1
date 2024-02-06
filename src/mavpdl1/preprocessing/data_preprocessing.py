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
