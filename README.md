# Med-AA

These are the benchmark experiments reported for the Med-AA dataset in our paper [A Dataset for Plain Language Adaptation of Answers to Consumer Health Questions](https://arxiv.org/pdf/2201.12888.pdf)

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
Run the following command:

```
python BaselineModelReports.py
```

## Creating Metrics
Run the following command:

```
python Metrics.py
```

That's it! Thank you for using this code, and please contact us if you find any issues with the repository or have questions about text adaptation. If you publish work related to this project, please cite
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
