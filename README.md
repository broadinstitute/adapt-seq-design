# adapt-seq-design
Methods for predicting sequence activity.

This repository contains training data and models used by ADAPT, as well as analyses of their performance.

**For more information on ADAPT and on how to run it, please see the [ADAPT repository](https://github.com/broadinstitute/adapt) on GitHub.**

## Summary of contents
Below is a summary of this repository's contents:
* `data/`: Library design and training/test data for CRISPR-Cas13a.
* `models/`: Saved classification and regression models, as determined by a hyperparameter search.
* `out/`: Results of Cas13a library data analyses, model hyperparameter searches, nested cross-validation, and learning curves.
* `plotting_scripts/`: Scripts to produce plots on the data in this repository.
* `model.sh`: Wrapper script for training and evaluating models; includes calls to the Python modules.
* `predictor.py`, `predictor_baseline.py`: Methods to train and evaluate classification and regression models.
* `parse_data.py`: Methods to parse our data and prepare them for training/testing (encoding, splits, sampling, etc.).
* `predictor_hyperparam_search.py`: Methods to perform a hyperparameter search over models.
* `predictor_learning_curve.py`: Methods to produce a learning curve for the models.
