# a transformer model for IOB NER sequential token classification
import torch

from transformers import BertForTokenClassification, TrainingArguments

import evaluate

seqeval = evaluate.load("seqeval")


class BERTNER:
    def __init__(self, config, label_encoder, tokenizer):
        self.model = BertForTokenClassification.from_pretrained(
            config.model, num_labels=len(label_encoder.classes_)
        )
        # resize the model bc we added emebeddings to the tokenizer
        self.model.resize_token_embeddings(len(tokenizer))

        # set tokenizer
        self.tokenizer = tokenizer

        # set label encoder
        self.label_encoder = label_encoder

        # set training args
        self.training_args = TrainingArguments(
            output_dir=config.savepath,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            evaluation_strategy=config.evaluation_strategy,
            save_strategy=config.save_strategy,
            load_best_model_at_end=config.load_best_model_at_end,
            warmup_steps=config.warmup_steps,
            logging_dir=config.logging_dir,
            dataloader_pin_memory=config.dataloader_pin_memory,
            metric_for_best_model=config.metric_for_best_model,
            weight_decay=config.weight_decay,
            use_cpu=config.use_cpu,
        )

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
