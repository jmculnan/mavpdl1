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
        "torch",
        "datasets",
        "transformers",
        "accelerate>=0.20.1",
        "scikit-learn",
        "seqeval",
        "evaluate",
        "urllib3==1.26.6",
        '"ray[tune]"'
    ]
)
