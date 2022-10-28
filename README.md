# PLABA

These are the benchmark experiments reported for the PLABA dataset in our manuscript "A Dataset for Plain Language Adaptation of Biomedical Abstracts"

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

Thank you for using this code. Please contact us if you find any issues with the repository or have questions about text adaptation.
