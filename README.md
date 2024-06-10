# EngineBench
[Project page](https://eng.ox.ac.uk/tpsrg/research/enginebench/) | [Link to paper](https://arxiv.org/abs/2406.03325)

This is the EngineBench database, a collection of datasets collated by the [Oxford TPSRG](https://eng.ox.ac.uk/tpsrg/) specially for machine learning research in thermal propulsion systems. EngineBench is comprised of Particle Image Velocimetry (PIV) data from different experiments previously run on the transparent combustion chamber (TCC-III) optical engine by General Motors and the University of Michigan. Code for inpainting problems is 

## Table of Contents
1. [Acknowledgements](#acknowledgements)
2. [Quickstart](#quickstart)
4. [Installation](#installation)
5. [Usage](#usage)

## Acknowledgements

Publications arising from the use of EngineBench should cite: 
1. The EngineBench database DOI
2. The following publication:
   ```bibtex
   @misc{baker2024enginebench,
      title={EngineBench: Flow Reconstruction in the Transparent Combustion Chamber III Optical Engine}, 
      author={Samuel J. Baker and Michael A. Hobley and Isabel Scherl and Xiaohang Fang and Felix C. P. Leach and Martin H. Davy},
      year={2024},
      eprint={2406.03325},
      archivePrefix={arXiv},
      primaryClass={physics.flu-dyn}
   }
3. Include the following acknowledgment to the original data source:
  
   "The TCC engine work has been funded by General Motors through the General Motors University of
   Michigan Automotive Cooperative Research Laboratory, Engine Systems Division."

This work uses neural network implementations from other sources:
* UNet and UNETR: [Project MONAI](https://github.com/Project-MONAI/MONAI)
* Context encoder GAN: [BoyuanJiang](https://github.com/BoyuanJiang/context_encoder_pytorch) (`model.py` file archived to `inpainting/external/` 20-05-2024)

## Quickstart
The quickest way to start using EngineBench is via our tutorials using Kaggle notebooks:
1. [Browse the data](https://www.kaggle.com/code/samueljbaker/browsedata)
2. [Test different gap types](https://www.kaggle.com/code/samueljbaker/gaptester) for inpainting
3. [Train an inpainting model](https://www.kaggle.com/code/samueljbaker/trainingexample)

## Installation 
### Requirements

- Python 3.8+
- See `environment.yml` for a full list of dependencies.

### Install

```bash
git clone https://github.com/...
cd EngineBench
conda env create -f environment.yml
```
## Usage
### Data Download
Download the dataset from one of the Kaggle repositories:
* [Small dataset](https://www.kaggle.com/datasets/samueljbaker/enginebench-lsp-small/)
* [Full dataset](https://www.kaggle.com/datasets/samueljbaker/enginebench)

### Training Configuration
Choose the training configurations by editing or creating a new `.yaml` file in the `inpainting/configs/` directory.

### Model Training
To train a model, use one of the following scripts based on the model type defined in your config file:
For a UNet or UNETR model:
```bash
python train.py --config configs/test_config.yaml
```
For a GAN model:
```bash
python train_gan.py --config configs/test_config.yaml
```

### Model evaluation
Evaluate either model using:
```bash
python test.py --config configs/test_config.yaml
```
