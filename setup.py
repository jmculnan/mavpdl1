# setup.py
import setuptools

setuptools.setup(
    name="mavpdl1",
    version="1.0",
    author="VA MAVERIC Informatics",
    description="Predicts PD-L1 values, test vendors, and units from notes",
    packages=[
            "src/mavpdl1"
    ],
    install_requires=[
        "pandas==2.2.0",
        "torch==2.0.0",
        "datasets==2.16.1",
        "transformers==4.30.2",
        "accelerate>=0.20.1",
        "scikit-learn==1.4.0",
        "seqeval==1.2.2",
        "evaluate==0.4.1",
        "urllib3==1.26.6",
        "optuna==3.5.0"
    ]
)
