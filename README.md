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
Then, run train normally from the base directory (`python scripts/train.py`)