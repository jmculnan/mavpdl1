# train an NER model

# IMPORT STATEMENTS
import evaluate
import pandas as pd
import torch
import logging

from transformers import Trainer, DataCollatorForTokenClassification
from datasets import Dataset

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix

from bert_sklearn import BertTokenClassifier
from optuna.integration import OptunaSearchCV
from optuna import distributions

from seqeval.metrics import classification_report

from config import train_config as config

# from other modules here
from mavpdl1.utils.utils import (
    get_from_indexes,
    CustomCallback,
    id_labeled_items,
    optuna_hp_space
)
from mavpdl1.preprocessing.data_preprocessing import PDL1Data
from mavpdl1.model.ner_model import BERTNER

# load seqeval in evaluate
seqeval = evaluate.load("seqeval")


if __name__ == "__main__":
    # PREPARE DATA
    # ---------------------------------------------------------
    # use deidentified data sample
    pdl1 = PDL1Data(
        config.dataset_location,
        model=config.model,
        ner_classes=["unit"],
        classification_classes=["test"],
        classification_type="multilabel",
    )
    all_data = pdl1.data

    # get tokenized data and IOB-2 gold labeled data
    tokenized, gold= pdl1.tokenize_label_data()

    # convert data to train, dev, and test
    # percentage breakdown and code formatting from Kyle code
    (
        X_train_full,
        X_test,
        y_train_full,
        y_test,
    ) = train_test_split(
        tokenized, gold, test_size=0.15, random_state=config.seed
    )

    # generate the test dataset if needed
    # train and val done below due to nature of the tasks
    test_dataset = Dataset.from_dict({"texts": X_test, "label": y_test})
    test_dataset = test_dataset.map(pdl1.tokenize, batched=True)

    # convert train data into KFold splits
    # splits = config.ner_num_splits
    # folds = KFold(n_splits=splits, random_state=config.seed, shuffle=True)
    # idxs = range(len(X_train_full))

    # create a holder for all items that you've IDed as
    # having a PD-L1 value
    all_labeled = []

    ner_model = BertTokenClassifier('scibert-scivocab-uncased', logfile=f"{config.savepath}/optuna.log")

    # set the possible search space
    # these names come from the sklearn-bert BertBaseEstimator class keywords
    param_distributions = {
        "learning_rate": distributions.FloatDistribution(1e-6, 1e-4, log=True),
        "train_batch_size": distributions.CategoricalDistribution([1, 2, 4]),
        "epochs": distributions.CategoricalDistribution([1, 2, 4])
    }

    # set optuna search cv
    optuna_search = OptunaSearchCV(ner_model, param_distributions, cv=5, n_trials=1)

    optuna_search.fit(X_train_full, y_train_full)

    logging.info("Best trial:")
    trial = optuna_search.study_.best_trial

    logging.info("  Value: ", trial.value)
    logging.info("  Params: ")
    for key, value in trial.params.items():
        logging.info(f"    {key}: {value}")

    exit()
    #
    # # do CV over the data
    # for i, (train_index, test_index) in enumerate(folds.split(idxs)):
    #     logging.info(f"Now beginning fold {i}")
    #
    #     X_train, X_val = get_from_indexes(X_train_full, train_index, test_index)
    #     y_train, y_val = get_from_indexes(y_train_full, train_index, test_index)
    #
    #     # todo: kyle code used `label` but on CPU would not train
    #     #   unless i changed it to `labels`
    #     train_dataset = Dataset.from_dict(
    #         {"texts": X_train, "labels": y_train}
    #     )
    #     train_dataset = train_dataset.map(pdl1.tokenize, batched=True)
    #
    #     val_dataset = Dataset.from_dict(
    #         {"texts": X_val, "labels": y_val}
    #     )
    #     val_dataset = val_dataset.map(pdl1.tokenize, batched=True)
    #
    #     # TRAIN THE MODEL
    #     # ---------------------------------------------------------
    #     # we need this to be a 2 part model
    #     # part 1: IOB NER task over the data to ID results; this will need to be trained as a first step
    #     # part 2: AFTER training the NER system, we need to select only those items IDed by the NER
    #     #   and use those in the classification task
    #
    #     # PART 1
    #     ner = BERTNER(config, pdl1.ner_encoder, pdl1.tokenizer)
    #
    #     # add data collator
    #     data_collator = DataCollatorForTokenClassification(
    #         pdl1.tokenizer, padding=True, return_tensors="pt"
    #     )
    #
    #     ner.update_save_path(f"{config.savepath}/ner/fold_{i}")
    #     ner.update_log_path(f"{config.ner_logging_dir}/fold_{i}")
    #
    #     # set up trainer
    #     trainer = Trainer(
    #         model=ner.model,
    #         args=ner.training_args,
    #         train_dataset=train_dataset,
    #         data_collator=data_collator,
    #         eval_dataset=val_dataset,
    #         compute_metrics=ner.compute_metrics,
    #     )
    #     # add callback to print train compute_metrics for train set
    #     # in addition to val set
    #     trainer.add_callback(CustomCallback(trainer))
    #
    #     # train the model
    #     metrics = trainer.train()
    #
    #     # get the best predictions from the model
    #     # to select data inputs for classification task
    #     y_pred = trainer.predict(val_dataset)
    #
    #     # IDENTIFY ALL SAMPLES THAT WILL MOVE ON TO BE USED AS INPUT WITH NEXT MODEL
    #     # define a second computation
    #     all_labeled_ids = id_labeled_items(
    #         y_pred.predictions,
    #         val_dataset["texts"],
    #         pdl1.ner_encoder.transform(["O"]),
    #     )
    #     all_labeled.extend(all_labeled_ids)
    #
    #     predictions = torch.argmax(torch.from_numpy(y_pred.predictions), dim=2)
    #     labels = [list(map(int, label)) for label in val_dataset["labels"]]
    #
    #     true_labels = [pdl1.ner_encoder.inverse_transform(label) for label in labels]
    #     true_predictions = [
    #         pdl1.ner_encoder.inverse_transform(prediction) for prediction in predictions
    #     ]
    #
    #     true_predictions = list(map(list, true_predictions))
    #     true_labels = list(map(list, true_labels))
    #
    #     preds_for_confusion = predictions.flatten().squeeze().tolist()
    #     labels_for_confusion = [label for label_list in labels for label in label_list]
    #
    #     logging.info(f"Confusion matrix on validation set: ")
    #     logging.info(confusion_matrix(labels_for_confusion, preds_for_confusion))
    #
    #     report = classification_report(true_labels, true_predictions, digits=2)
    #     logging.info(pdl1.ner_encoder.classes_)
    #
    #     logging.info(report)
    #
    # # save the list of all documents containing PD-L1 values according to the model
    # labeled_df = pd.DataFrame(all_labeled, columns=["CANDIDATE"])
    # labeled_df.to_csv(
    #     f"{config.savepath}/all_documents_with_IDed_pdl1_values.csv", index=False
    # )
