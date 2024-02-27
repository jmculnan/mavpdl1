# Test project used for NER task on PD-L1 values, vendors, and units

To use this project's code, first create a virtual environment in the base directory (ner_test):

```python3 -m venv <venv_name>```

Activate the new enviroment: 

```source <path/to/venv>/bin/activate```

After ensuring that your environment is activated, install the necessary libraries and packages with: 

```pip install .``` 

or, if you want the editable version:

```pip install -e .```

## To train a model

First, update the file paths in `config/train_config.py` 
(currently the only config file; add inference config later as needed)
You will probably want to run hyperparameter tuning on the models.
To do this, run `python scripts/train_hptuning.py` from the base directory.

If you only want to complete hyperparameter tuning on one portion of the model, use the relevant script: 
`scripts/train_classifier_only.py` or `scripts/train_ner_only_hptuning.py`. 

If you want to run training without hyperparameter tuning, run (`python scripts/train.py`) from the base directory.
