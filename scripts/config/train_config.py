# config file with relevant paths for testing NER system
import logging
import sys
from pathlib import Path

# set random seed
seed = 42

# set savepath
savepath = "output/testing"
Path(savepath).mkdir(parents=True, exist_ok=True)

# whether to save plots
save_plots = True
# whether to save the trained model
save_model = True

# set output info
outname = "first_file"
experiment_no = "1"
expname = f"{experiment_no}_{outname}"

# dataset used
dataset_location = "/Users/jculnan/va_data/pdl1_annotations-100_deidentified_fakeuuids.csv"
# dataset_location = "/Users/jculnan/va_data/pdl1_annotations-10_deidentified.csv"
# model
model = "allenai/scibert_scivocab_uncased"

# decide if you want logging to go to stdout or stderr
# in addition to going to a save file
# modified from https://stackoverflow.com/questions/13733552/logger-configuration-to-log-to-file-and-print-to-stdout
use_stdout = True
# create logger
# formatting the logger
logFormatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s")
# make a root logger to hold the two others
rootLogger = logging.getLogger()
# add logger that saves to file
fileHandler = logging.FileHandler(f"{savepath}/{expname}.log")
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)
# add logger that directs to stdout
if use_stdout:
    consoleHandler = logging.StreamHandler(sys.stdout)
else:
    consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)


# model parameters
# todo: make separate args for NER model and classifier
num_epochs = 4  # 50
per_device_train_batch_size = 4  # 32
per_device_eval_batch_size = 64
evaluation_strategy = "epoch"
save_strategy = "epoch"
load_best_model_at_end = True
warmup_steps = 500
logging_dir = 'output/logs'
dataloader_pin_memory = False
metric_for_best_model = 'f1'
weight_decay = 0.001
use_cpu = True

# ner model-specific parameters
num_splits = 5
