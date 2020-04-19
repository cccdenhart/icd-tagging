# ICD Code Tagging

This project is an attempt to implement the task of [ICD code tagging](https://arxiv.org/pdf/1802.02311.pdf) with modern machine learning techniques. 

## Attribution

Much of this project was inspired by the following papers:
- [An Empirical Evaluation of Deep Learning for ICD-9 CodeAssignment using MIMIC-III Clinical Notes](https://arxiv.org/pdf/1802.02311.pdf)
- [Tagging Patient Notes With ICD-9 Codes](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/reports/2744196.pdf)

Additionally, ALL code in the `icd9/` folder was directly cloned from [this](https://github.com/sirrice/icd9) repository. This code is used for navigating the ICD9 code hierarchy via python objects. It was not available from a package manager, so it was incorporated directly to the project.

## Project structure

The `scripts/` directory contains python files with code for frontloading tasks that are cumbersome/expensive. 

The `build.py` file acts as the controller for the above tasks, accomodating command-line interaction.  This file can also be executed through the `Makefile` as described in the below section.

The `report.ipynb` file contains all code for exploratory/evaluation visualizations.

## Reproducing code

In order to run any of this code, you must first obtain access to [MIMIC-III](https://github.com/MIT-LCP/mimic-code) and deploy the dataset to your AWS account.

Run `pipenv install` to install all dependencies.

You should create a `.env` file in the project root. AWS credentials should be stored here in the following variables:
- ACCESS_KEY
- SECRET_KEY
- S3_DIR
- REGION_NAME

Prerprocessing and model training can be conducted through the Makefile as follows:

- `make preprocess` to conduct all preprocessing steps.
- `make split` to split the data into train/test data
- `make train` to train models 

Once this is all complete, you should be able to execute all cells in the `report.ipynb`. 
