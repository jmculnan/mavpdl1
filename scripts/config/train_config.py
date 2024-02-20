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
# whether to evaluate on the test set at end of training
evaluate_on_test_set = False

# set output info
outname = "first_file"
experiment_no = "1"
expname = f"{experiment_no}_{outname}"

# provide list of paths to data documents
dataset_location = ["/Users/jculnan/va_data/pdl1_annotations-100_deidentified_fakeuuids.csv"]
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
rootLogger.setLevel(logging.INFO)
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

######################################################
############### SHARED PARAMETERS ####################
######################################################
# these parameters are used by BOTH the NER model and
# the document-level classifier
evaluation_strategy = "epoch"
save_strategy = "epoch"
total_saved_epochs = 1  # only save best model
logging_strategy = 'epoch'
load_best_model_at_end = True
dataloader_pin_memory = False
metric_for_best_model = 'f1'
use_cpu = True
warmup_steps = 500

######################################################
############# NER MODEL PARAMETERS ###################
######################################################
# these parameters will be used with the NER model
# but NOT with the document-level classifier
ner_num_epochs = 4  # 50
ner_per_device_train_batch_size = 4  # 32
ner_per_device_eval_batch_size = 64
ner_logging_dir = f'{savepath}/ner/logs'
ner_weight_decay = 0.001
ner_num_splits = 5

######################################################
############# CLS MODEL PARAMETERS ###################
######################################################
# these parameters will be used with the document
# classifier but NOT with the NER model
cls_num_epochs = 4  # 50
cls_per_device_train_batch_size = 4  # 32
cls_per_device_eval_batch_size = 64
cls_logging_dir = f'{savepath}/classifier/logs'
cls_weight_decay = 0.001
