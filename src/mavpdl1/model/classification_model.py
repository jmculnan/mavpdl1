# a transformer model for classification at the text level
import torch
import numpy as np
import logging

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import TrainingArguments, BertForSequenceClassification


class BERTTextMultilabelClassifier:
    """
    A BERT-based document-level multilabel classifier
    """
    def __init__(self, config, label_encoder, tokenizer, model=None):
        # we may want to be able to provide a model we've been training with
        # as part of a longer training procedure
        if model:
            self.model = model
        else:
            self.model = BertForSequenceClassification.from_pretrained(
                config.model,
                num_labels=len(label_encoder.classes_),
                problem_type="multi_label_classification",
            )
        # resize the model bc we added emebeddings to the tokenizer
        self.model.resize_token_embeddings(len(tokenizer))

        # put device on gpu or cpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        self.model.to(self.device)

        # set tokenizer
        self.tokenizer = tokenizer

        # set label encoder
        self.label_encoder = label_encoder

        # set training args
        self.training_args = TrainingArguments(
            output_dir=f"{config.savepath}/classifier",
            num_train_epochs=config.cls_num_epochs,
            per_device_train_batch_size=config.cls_per_device_train_batch_size,
            per_device_eval_batch_size=config.cls_per_device_eval_batch_size,
            evaluation_strategy=config.evaluation_strategy,
            save_strategy=config.save_strategy,
            load_best_model_at_end=config.load_best_model_at_end,
            warmup_steps=config.warmup_steps,
            logging_dir=config.cls_logging_dir,
            dataloader_pin_memory=config.dataloader_pin_memory,
            metric_for_best_model=config.metric_for_best_model,
            weight_decay=config.cls_weight_decay,
            use_mps_device=True if self.device == torch.device('mps') else False,
        )

    def multilabel_compute_metrics(self, pred_targets):
        """
        :param pred_targets: An instance of class transformers.EvalPrediction;
            consists of an np.ndarray of predictions, an np.ndarray of targets,
            and an optional np.ndarray of inputs
        :return: precision, recall, f1, accuracy

        todo: the metrics used depend on task formulation
            -- multitask (ID vendor + unit separately, 1 item per example)
            -- singletask multilabel (vendor + unit together, 1+ per example)
            -- multitask multilabel (vendor + unit separately, 1+ per example)
        """
        predictions, targets = pred_targets

        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        # next, use threshold to turn them into integer predictions
        preds = np.zeros(probs.shape)
        # use 0.5 as a threshold for predictions
        preds[np.where(probs >= 0.5)] = 1

        # convert targets to ints
        targets = [list(map(int, label)) for label in targets]

        # calculate precision, recall, f1, support
        # selected macro f1 for imbalanced classes
        # can change as needed
        results = precision_recall_fscore_support(targets, preds, average='macro',
                                                  zero_division=0.0)
        accuracy = accuracy_score(targets, preds)

        # todo: this doesn't indicate whether you're getting results on
        #   train or evaluation data -- will need to alter
        logging.info(results)

        return {
            "precision": results[0],
            "recall": results[1],
            "f1": results[2],
            "accuracy": accuracy,
        }
