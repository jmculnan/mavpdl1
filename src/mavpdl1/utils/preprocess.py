import pandas as pd
import re
from pathlib import Path


def convert_all_files(path_to_base_dir):
    """
    Convert all corresponding files within
    :param path_to_base_dir:
    :return:
    """
    all_data = []
    ann = Path(f"{path_to_base_dir}/original")
    texts = Path(f"{path_to_base_dir}/text")
    for ann_file in ann.iterdir():
        ann_name = ann_file.stem
        full_texts = texts / f"{ann_name}.txt"
        short_df = convert_file_to_bio(ann_file, full_texts)
        if short_df is not None:
            all_data.append(short_df)

    return pd.concat(all_data)


def convert_file_to_bio(ann_file_path, text_file_path):
    """
    Convert an annotation file + text file to the proper
    format to be used by a transformer model for NER
    :param ann_file_path: the path to an annotation file
    :param text_file_path: the path to a raw text file
    :return: a pandas DF in BIO format
    """
    ann_data = pd.read_csv(
        ann_file_path, sep="\t", names=["entity_num", "type", "tokens"]
    )
    if len(ann_data) == 0:
        return None

    # got only total onset and offset character numbers
    # this might not even be necessary
    # if it isn't you can simply remove all data after the first space in this col
    ann_data["type"] = ann_data["type"].apply(
        lambda x: re.sub(r"[0-9]+;[0-9]+\s", "", x)
    )

    ann_data[["type", "onset", "offset"]] = ann_data["type"].str.split(" ", expand=True)

    # remove AnnotatorNotes rows
    ann_data = ann_data[ann_data["type"] != "AnnotatorNotes"]

    # place each token on a separate line to add B- and I-
    ann_data["tokens"] = ann_data["tokens"].str.split(" ", expand=False)
    ann_data = ann_data.explode("tokens")

    # save the type and token to lists
    types = ann_data["type"].tolist()
    tokens = ann_data["tokens"].tolist()
    onsets = ann_data["onset"].tolist()

    # convert these to B + I + type
    types = convert_to_iob(types, onsets)

    # each line in text file corresponds to a single sentence
    ordered_wds = []
    with open(text_file_path, "r") as tf:
        for line in tf:
            # remove punctuation from lines
            line = re.sub(r"[^\w\s]", "", line)
            ordered_wds.append(line.strip().split(" "))

    # combine these
    all_tokens = []
    all_types = []
    c = 0
    for wds in ordered_wds:
        for wd in wds:
            if c < len(tokens) and wd == tokens[c]:
                all_tokens.append(wd)
                all_types.append(types[c])
                c += 1
            else:
                all_tokens.append(wd)
                all_types.append("O")

    types_tokens = {"gold": all_types, "token": all_tokens, "file": text_file_path.stem}

    corrected_data = pd.DataFrame(types_tokens)
    return corrected_data


def convert_to_iob(types, onsets):
    """
    Convert types to IOB types
    :param types:
    :param onsets:
    :return:
    """
    # handle the first item; always B
    types[0] = "B-" + types[0]

    # for all after the first item
    for i in range(1, len(onsets)):
        if onsets[i] == onsets[i - 1]:
            types[i] = "I-" + types[i]
        else:
            types[i] = "B-" + types[i]

    return types


if __name__ == "__main__":
    from sklearn.preprocessing import LabelEncoder

    data_path = "/Users/jculnan/Downloads/cadec/"

    # use cadec dataset for this test
    # if we get to do more testing later, replace with real data
    all_data = convert_all_files(data_path)

    # get label set
    label_set = all_data["gold"].unique()

    # encoder for labels
    label_encoder = LabelEncoder()
    label_encoder.fit(label_set)

    # convert this df so that you have lists of tokens and gold labels for making data splits
    data_as_list = (
        all_data.groupby(["file"])[["gold", "token"]]
        .agg(list)
        .reset_index()
        .reindex(all_data.columns, axis=1)
    )

    # subsequent steps as in train script
