# import statements
import pandas as pd
import torch
import json
import evaluate
import logging

from transformers import (
    BertForTokenClassification,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from seqeval.scheme import IOB1
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)


# read in data


# convert data to train, dev, and test


# either read in trained model or train the model


# generate predictions on test partition
