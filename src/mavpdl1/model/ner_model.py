# a transformer model for IOB NER sequential token classification
import torch
import logging
import evaluate

from transformers import BertForTokenClassification, TrainingArguments

seqeval = evaluate.load("seqeval")


class BERTNER:
    """
    A BERT-based model for NER classification
    """

    def __init__(self, config, label_encoder, tokenizer):
        self.model = BertForTokenClassification.from_pretrained(
            config.model, num_labels=len(label_encoder.classes_)
        )
        # resize the model bc we added emebeddings to the tokenizer
        self.model.resize_token_embeddings(len(tokenizer))

        # put device on gpu or cpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
        self.model.to(self.device)

        # set tokenizer
        self.tokenizer = tokenizer

        # set label encoder
        self.label_encoder = label_encoder

        # set config
        self.config = config

        # set training args
        self.training_args = TrainingArguments(
            output_dir=config.savepath,
            num_train_epochs=config.ner_num_epochs,
            per_device_train_batch_size=config.ner_per_device_train_batch_size,
            per_device_eval_batch_size=config.ner_per_device_eval_batch_size,
            evaluation_strategy=config.evaluation_strategy,
            save_strategy=config.save_strategy,
            save_total_limit=config.total_saved_epochs,
            load_best_model_at_end=config.load_best_model_at_end,
            warmup_steps=config.warmup_steps,
            logging_dir=config.ner_logging_dir,
            logging_strategy=config.logging_strategy,
            dataloader_pin_memory=config.dataloader_pin_memory,
            metric_for_best_model=config.metric_for_best_model,
            weight_decay=config.ner_weight_decay,
            use_mps_device=True if self.device == torch.device("mps") else False,
        )

    def reinit_model(self):
        self.model = BertForTokenClassification.from_pretrained(
            self.config.model,
            num_labels=len(self.label_encoder.classes_),
        )
        # resize the model bc we added emebeddings to the tokenizer
        self.model.resize_token_embeddings(len(self.tokenizer))

        return self.model

    def compute_metrics(self, pred_targets):
        """
        From Kyle code
        :param pred_targets: An instance of class transformers.EvalPrediction;
            consists of an np.ndarray of predictions, an np.ndarray of targets,
            and an optional np.ndarray of inputs
        :return: precision, recall, f1, accuracy
        """
        preds, targets = pred_targets
        # convert targets to ints
        targets = [list(map(int, label)) for label in targets]
        # get max prediction over classes for each prediction
        preds = torch.argmax(torch.from_numpy(preds), dim=2)
        # convert targets and preds to IOB1 + mentions
        # as this is what it uses to compute results
        true_predictions = [
            self.label_encoder.inverse_transform(prediction) for prediction in preds
        ]
        true_labels = [self.label_encoder.inverse_transform(label) for label in targets]

        # get results and return them
        results = seqeval.compute(predictions=true_predictions, references=true_labels)

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def load_best_hyperparameters(self, best_hyperparams):
        """
        After performing hyperparameter search with optuna,
        take the best hyperparams and update the trainingargs
        to reflect them
        :param best_hyperparams:
        :return:
        """
        for param in best_hyperparams.keys():
            try:
                self.training_args.param = best_hyperparams[param]
            except KeyError:
                logging.error(f"Unknown hyperparameter {param} listed")

    def update_save_path(self, new_path):
        self.training_args.output_dir = new_path

    def update_log_path(self, new_path):
        self.training_args.logging_dir = new_path
