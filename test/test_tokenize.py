# test tokenization as you develop it

import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv("/Users/jculnan/va_data/pdl1_annotations-100_deidentified.csv")
    data = data.dropna(subset=["LABELS"])
    data = data[["LABELS", "START", "END", "ANNOTATION_INDEX", "ANNOTATION", "CANDIDATE"]]

    row1 = data.iloc[1]
    test_sample = row1["CANDIDATE"]
    print(type(test_sample))
    test_start = row1["START"]
    test_end = row1["END"]
    test_ann = row1["ANNOTATION"]
    start_end_ann = test_sample[int(test_start):int(test_end)]

    c = 0
    for i, row in data.iterrows():
        sample = row["CANDIDATE"]
        ann = row["ANNOTATION"]
        extracted_ann = sample[int(row["START"]):int(row["END"])]
        if ann != extracted_ann:
            start = sample.find(ann)
            print(sample.find(ann), "\t", int(row["START"]))
            end = start + len(ann)
            new_extr = sample[start:end]
            if ann != new_extr:
                c = 1

    if c == 0:
        print("ALL EQUAL")
    else:
        print("SOMETHING IS OFF")

    # print(test_ann)
    # print("+++++++++++++++++")
    # print(start_end_ann)