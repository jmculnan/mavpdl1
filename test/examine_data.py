# examine the data stored in data/pdl1_annotations-100_deidentified.csv
import random

import pandas as pd
from random import sample
random.seed(20)

if __name__ == "__main__":
    dpath = "/Users/jculnan/va_data/pdl1_annotations-100_deidentified.csv"

    data = pd.read_csv(dpath)

    # get only data with labels
    labeled_data = data.dropna(subset=["LABELS"])

    # how many items have labels
    print(len(labeled_data))

    # get set of labels
    print(labeled_data.LABELS.unique())

    # get breakdown of labels by type
    print(labeled_data.LABELS.value_counts())

    # how many of these are actually distinct?
    distinct = labeled_data.CANDIDATE.unique()
    print(f"THERE ARE {len(distinct)} EXAMPLES HERE")
    # how many of overall candidates are distinct
    all_distinct = data.CANDIDATE.unique()
    print(f"THERE ARE {len(all_distinct)} EXAMPLES OVERALL")

    # look only at UT_UU data
    ut_uu = labeled_data[labeled_data.LABELS == "UT_UU"]
    print(ut_uu.iloc[0]["CANDIDATE"])
    print("=================================")

    # look only at dat with both test and unit
    d22c3_tps = labeled_data[labeled_data.LABELS == "D22C3_TPS"]
    print(d22c3_tps)  # the ANNOTATION column seems to be for the previous PD-L1 paper
    print(d22c3_tps.iloc[0]["CANDIDATE"])
    print("=================================")

    # have a look at a few of the items without labels
    unlabeled = data.loc[data.LABELS.isnull()]
    unlabeled_samples = unlabeled.CANDIDATE.unique().tolist()
    s1, s2, s3 = sample(unlabeled_samples, 3)
    print(s1)
    print("=================================")
    print(s2)
    print("=================================")
    print(s3)
    print("=================================")

    # save all candidates to a separate txt file for ease of reading
    with open("output/all_deidentified_candidates.txt", 'w') as tf:
        tf.write("\n".join(all_distinct.tolist()))

    with open("output/nonlabeled_deidentified_candidates.txt", 'w') as tf:
        tf.write("\n".join(unlabeled_samples))

    with open("output/labeled_deidentified_candidates.txt", 'w') as tf:
        tf.write("\n".join(distinct.tolist()))


