# [ICCV 2025] Region-Level Data Attribution for Text-to-Image Generative Models

Trong-Bang Nguyen, Phi-Le Nguyen, Simon Lucey, & Minh Hoai

Official PyTorch implementation for Attribution Region Detector (ARD). Details can be found in the paper, [[Paper]]().

If you find this repository useful, please give it a star ⭐.

## ARD Architecture

<img src=repo_img/architecture.png width="100%"/>

## Contents
* [Preparation](#preparation)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)

## Preparation

### Setup

We conduct our model running with the following settings: Python 3.9.19, and CUDA 12.1. It is possible that other versions are also available.

1. Clone this repository.

```bash
git clone https://github.com/bangdabezt/AR-Detector.git 
cd AR-Detector/
```

2. Install the required dependencies.

```bash
conda env create -f environment.yml
```

## Citation

### Acknowledgements

This repository is based on the [CountGD](https://github.com/niki-amini-naieni/CountGD/tree/main) and [Open-GroundingDino](https://github.com/longzw1997/Open-GroundingDino/tree/main), which are both built on [GroundingDINO repository](https://github.com/IDEA-Research/GroundingDINO).
