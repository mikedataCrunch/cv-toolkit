# cv-toolkit-tf

This repository serves as a standard development toolkit (SDK) for the development of basic DNN models for computer vision tasks. The SDK implements `tensorflow`,`keras`, and `pytorch` frameworks in `python`.


The tasks covered currently are:

1. Binary classification
2. Multi-class classficiation
3. Multi-label classification

## Set up

The coding environment used in the development of the SDK is set up using `conda` as such, it is expected that conda and dependencies are installed prior to using the toolkit.

To install the necessary libraries, run the following in the terminal:
```
conda env create -f environment.yml
```
As the SDK is currently underdevelopment, you may install the `cvtoolkit` *version 0.1* package after cloning by running below script within the top-level dir of `cv-toolkit` repo.

```
pip install -e .
```

This should allow imports within the environment. For example:
```
from cvtoolkit import explore
from cvtoolkit.tensorflow import train
```

## Data

Sample datasets are stored in the `datasets` directory. It is expected that datasets are structured following the sample tree below:

```
- datasets # root
	- <name-of-dataset1> # e.g., kvasir-capsule
		- <name-of-class1> # e.g., polyp
			- <image_id1> # e.g., patientid_frameid.jpg
			- <image_id2>
		- <name-of-class2>
		- <name-of-class3>
	- <name-of-dataset2>
    - <name-of-dataset3>
```

Additional metadata are in the `metadata` directory which contain dataframes (.csvs) of accompanying metadata indexed by image IDs (e.g., patientID_frameID). The filenames map to the files in the `datasets` diretory.

## Utilities

The scripts are contained in `utils` and are described as follows.

1. `src/*/binary.py` contains code for training binary classifiers
2. `src/*/multiclass.py` contains code for training multiclass classfiers.
3. `src/*/multilabel.py` contains code for training multilabellers
4. `src/*/inference.py` accepts a path to a saved model and an a dataset for inference and returns a `csv` that maps an image id to prediction probabilty.
5. `src/*/evaluate.py` accepts a path to a saved model and a validation or test dataset and outputs the following:
	- `figures/loss-curves.png`
	- `figures/confusion-matrix.png` (based on top-1 softmax for multiclass)
	- `figures/roc-pr-curves.png` (for binary classification)
6. `src/explore.py` generates an initial exploration of the dataset at:
	- `figures/samples.py` a labelled grid of randomly sampled images per class or class combination (i.e., in multilabel scenario).
7. `src/config` is a high level directory of configuration files needed to support the explore, train, inference, and evaluate scripts.


