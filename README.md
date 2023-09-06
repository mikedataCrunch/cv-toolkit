# cv-toolkit

This repository serves as a standard development toolkit (SDK) for the development of basic DNN models for computer vision tasks. The SDK implements `tensorflow` and `keras` frameworksin `python`.


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
	- <name of dataset2>
```

Additional metadata are in the `metadata` directory which contains dataframes (.csvs) of accompanying metadata index by image IDs (e.g., patientid_frameid). The filenames map to the same directory name in the `datasets` diretory.

## Utilities

The scripts are contained in `utils` and are described as follows.

1. `utils/binary.py` contains code for training binary classifiers
2. `utils/multiclass.py` contains code for training multiclass classfiers.
3. `utils/multilabel.py` contains code for training multilabellers
4. `utils/inference.py` accepts a path to a saved model and an a dataset for inference and returns a `csv` that maps an image id to prediction probabilty.
5. `utils/evaluate.py` accepts a path to a saved model and a validation or test dataset and outputs the following:
	- `figures/loss-curves.png`
	- `figures/confusion-matrix.png` (based on top-1 softmax for multiclass)
	- `figures/roc-pr-curves.png` (for binary classification)
6. `utils/explore.py` generates an initial exploration of the dataset at:
	- `figures/samples.py` a labelled grid of randomly sampled images per class or class combination (i.e., in multilabel scenario).
7. `utils/config` is a high level directory of configuration files needed to support the explore, train, inference, and evaluate scripts.


