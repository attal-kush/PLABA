# CABA

These are the benchmark experiments reported for the CABA dataset in our paper [A Dataset for Plain Language Adaptation of Answers to Consumer Health Questions](https://arxiv.org/pdf/2201.12888.pdf)

This repository contains the code to pre-process the data and run the text adaptation models presented in the paper.
If you are interested in just downloading the data, please refer to [OSF Link]. However, if you are interested in repeating the experiments reported in the paper, clone this repository, create a folder called 'data', and move the data found at https://doi.org/10.17605/OSF.IO/FYG46 to the 'data' folder/directory.

Please install [Anaconda](https://www.anaconda.com/distribution/) to create a conda environment as follows before Data Preparation, Training and Testing, or Metrics:
```shell script
# preparing environment
conda create -n medaa python=3.9
conda activate medaa
pip install --upgrade pip
pip install -r requirements.txt
```

## Data Preparation
Download the MedAA dataset from [OSF repository](https://doi.org/10.17605/OSF.IO/PC594) and place data.json in the `data` directory

## Training and Testing Benchmark Models
In the models directory, there are four deep learning systems:

* T5
* PEGASUS
* BART
* BART-LARGE-CNN

Running the baselines is simple and can be done while the medaa environment is active. Run the following command:

```
python BaselineModelReports.py
```
This will finetune the baseline models reported in the paper

## Creating Metrics

Once the models are trained and the baselines have been run on the dataset you are interested in evaluating, activate the medaa environment again. 
Then, run the following command:

```
python Metrics.py
```
More details about the metrics - BLEU, ROUGE, SARI - are described in the script. This script will generate the statistics reported in the paper with more technical detail.

Thank you for using this code. Please contact us if you find any issues with the repository or have questions about text adaptation. If you publish work related to this project, please cite
```
@article{attaladapt,
    title={A Dataset for Plain Language Adaptation of Answers to Consumer Health Questions},
    author={Kush Attal and Brian Ondov and Dina Demner{-}Fushman},
    journal = {arXiv e-prints}, 
    month = {[May]},
    year={[2020]},
    eprint={[2005.09067]},
    archivePrefix={arXiv},
    primaryClass={[cs.CL]}
    url={[https://arxiv.org/abs/2005.09067]}
}
