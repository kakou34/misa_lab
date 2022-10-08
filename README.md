# Medical Image Segmentation
## MAIA Master 2022


This repository contains all the code produced ass part of the Medical Image Segmetation (CAD) course part of the MAIA master. 

Laboratories:
    - Lab 1: SPM usage on brain tissue segmentations. (MATLAB based buuuu!)
    - Lab 2: Development of expectation maximization algorithms.

## Set up

Start by creating a new conda environment

```bash
conda update -y -n base -c defaults misa &&
conda create -y -n misa anaconda &&
conda activate misa
```

Install requirements:

```bash
pip install -r requirements.txt
```

### Download and prepare the database

```bash
mkdir data && mv data &&
gdown https://drive.google.com/uc?id=1JcaAFfX297Ui2o1FLz8F_Fw_tjPWukT3 &&
mkdir P2_data && mv P2_data.tar.gz P2_data/P2_data.tar.gz &&
cd P2_data/ && tar zxvf 'P2_data.tar.gz' && rm -rf 'P2_data.tar.gz' &&
cd ../..
```

#### Suggestions for contributers

- numpy docstring format
- flake8 lintern
- useful VSCode extensions:
  - autoDocstring
  - Python Docstring Generator
  - GitLens
