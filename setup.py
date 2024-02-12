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
        "pandas",
        "modin",
        "torch==2.0.0",
        "datasets",
        "transformers==4.30.2",
        "accelerate>=0.20.1",
        "scikit-learn",
        "seqeval",
        "evaluate",
        "urllib3==1.26.6",
        "optuna"
    ]
)
