# [ICCV 2025] Region-Level Data Attribution for Text-to-Image Generative Models

Official PyTorch implementation for Attribution Region Detector (ARD). Details can be found in the paper, [[Paper]]().

If you find this repository useful, please give it a star ⭐.

> **Region-Level Data Attribution for Text-to-Image Generative Models**<br>

> Trong-Bang Nguyen, Phi-Le Nguyen, Simon Lucey, and Minh Hoai <br>
> ICCV 2025
>
>**Abstract**: <br>
Data attribution in text-to-image generative models is a crucial yet underexplored problem, particularly at the regional level, where identifying the most influential training regions for generated content can enhance transparency, copyright protection, and error diagnosis. Existing data attribution methods either operate at the whole-image level or lack scalability for large-scale generative models. In this work, we propose a novel framework for region-level data attribution. At its core is the Attribution Region (AR) detector, which localizes influential regions in training images used by the text-to-image generative model. To support this research, we construct a large-scale synthetic dataset with ground-truth region-level attributions, enabling both training and evaluation of our method. Empirical results show that our method outperforms existing attribution techniques in accurately tracing generated content back to training data. Additionally, we demonstrate practical applications, including identifying artifacts in generated images and suggesting improved replacements for generated content. Our dataset and framework will be released to advance further research in region-level data attribution for generative models.

---

</div>

## AR-Detector Architecture

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
conda activate AR_Detector
```

## Citation

### Acknowledgements

This repository is based on the [CountGD](https://github.com/niki-amini-naieni/CountGD/tree/main) and [Open-GroundingDino](https://github.com/longzw1997/Open-GroundingDino/tree/main), which are both built on [GroundingDINO repository](https://github.com/IDEA-Research/GroundingDINO).
