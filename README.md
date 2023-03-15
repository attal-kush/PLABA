# PLABA

These are the benchmark experiments reported for the PLABA dataset in our Nature Scientific Data paper "A Dataset for Plain Language Adaptation of Biomedical Abstracts"

This repository contains the code to pre-process the data and run the text adaptation models presented in the paper.
If you are interested in just downloading the data, please refer to [https://osf.io/rnpmf/](https://osf.io/rnpmf/). However, if you are interested in repeating the experiments reported in the paper, clone this repository, create a folder called 'data', and move the data found at https://osf.io/rnpmf/ to the 'data' folder/directory.

Please install [Anaconda](https://www.anaconda.com/distribution/) to create a conda environment as follows before Data Preparation, Training and Testing, or Metrics:
```shell script
# preparing environment
conda create -n plaba python=3.9
conda activate plaba
pip install --upgrade pip
pip install -r requirements.txt
```

## Data Preparation
Download the PLABA dataset from [https://osf.io/rnpmf/](https://osf.io/rnpmf/) and place data.json and the CSV files in the `data` directory

## Training and Testing Benchmark Models
In the models directory, there are six deep learning systems:

* T5
* PEGASUS
* BART
* BART-LARGE-CNN
* T5 (without pretraining)
* T0PP (without pretraining)

Running the baselines is simple and can be done while the plaba environment is active. Run the following command:

```
python BaselineModelReports.py
```
This will finetune the baseline models reported in the paper

## Creating Metrics

Once the models are trained and the baselines have been run on the dataset you are interested in evaluating, activate the plaba environment again. 
Then, run the following command:

```
python Metrics.py
```
More details about the metrics - BLEU, ROUGE, SARI, BERTSCORE - are described in the script. This script will generate the statistics reported in the paper with more technical detail.

## Training and Testing Additional Seq2Seq Models

If you choose to replace one of the benchmark models and train and test a different deep learning model on this dataset (e.g., SciFive, BioBART), it can be performed in 2 steps:

1. Replace the [huggingface](https://huggingface.co/models) reference for the old deep learning model with the reference for the new model. For example, if you choose to replace the T0PP model with SciFive, replace "bigscience/T0_3B" in lines 321, 325, and 329 of BaselineModelReports.py with "razent/SciFive-large-Pubmed_PMC". You can also change the batch_size, max_token_length, and other parameters if needed to match the default parameters of the new model.

2. Replace all mentions of the old model in-code with the new model. For example, if replacing T0PP with SciFive, replace all mentions of T0PP with SciFive in BaselineModelReports.py (i.e., variable names, dictionary keys, etc.). This step is for labelling purposes and does not affect the actual model-generated adaptations.

Thank you for using this code. Please contact us if you find any issues with the repository or have questions about text adaptation. If you plublish work related to this project, please cite 
```
@article{attal_dataset_2023,
	title = {A dataset for plain language adaptation of biomedical abstracts},
	volume = {10},
	issn = {2052-4463},
	url = {https://doi.org/10.1038/s41597-022-01920-3},
	doi = {10.1038/s41597-022-01920-3},
	number = {1},
	journal = {Scientific Data},
	author = {Attal, Kush and Ondov, Brian and Demner-Fushman, Dina},
	month = jan,
	year = {2023},
	pages = {8},
}
```
