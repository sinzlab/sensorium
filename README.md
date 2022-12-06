# SENSORIUM 2022 Competition Submission

<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![hub](https://img.shields.io/badge/powered%20by-hub%20-ff5a1f.svg)](https://github.com/activeloopai/Hub)

## Contents

1. [Overview](#1-overview)
2. [Setup Instructions](#2-setup-instructions)
3. [Experiments](#3-experiments)

## 1. Overview

![Fig1](https://user-images.githubusercontent.com/102295389/205992650-ce6d5b0f-99b6-4e25-b88d-7cc7a4124925.png)

This repo contains our submission for the **NeurIPS 2022 The SENSORIUM competition** on predicting large scale mouse primary visual cortex activity.<br/>
The competition aimed to find the best neural predictive model that can predict the activity of thousands of neurons in the primary visual cortex of mice in response to natural images.

### Tracks
**SENSORIUM** - Stimulus-only - Assessed on how well they predict neural activity solely considering the stimulus averaged over trials.<br/>
**SENSORIUM+** - Stimulus-and-Behavior - Assessed based on how well they can predict individual trials given behavioral variables.

## 2. Setup Instructions

- Clone the repo:

    ```.bash
    git clone https://github.com/praeclarumjj3/NST-Tech.git
    cd NST-Tech
    ```

- Create a conda environment:

    ```.bash
    conda env create -f conda_env.yml
    conda activate nst
    ```

## 3. Experiments

### Training

- Execute the following command to run style transfer:

    ```bash
    sh nst.sh
    ```

>Note: There are arguments specified in the `nst.sh` script. Please modify them to run experiments under different settings.
- You may specify the `content` and `style` images to be used from the [data/content](data/content) and [data/style](data/style) folders respectively.

### Evaluation

- We use the predictions from the [AdaIn-Style](https://github.com/xunhuang1995/AdaIN-style) method proposed in [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868) as ground truths while evaluating the performance of our method.

- Install [`image-similarity-measures`](https://github.com/up42/image-similarity-measures):

    ```.bash
    pip install image-similarity-measures[speedups]
    ```

- Execute the following command to calculate the `PSNR`, `SSIM` and `RMSE` scores:

    ```.bash
    sh metrics.sh [path-to-gt] [path-to-our-prediction]
    ```

>Note: The gts can be found in the [`data/gts/`](data/gts/) directory. You may specify more metrics in the [metrics.sh](metrics.sh) script.
## Acknowledgement

This repo is a part of our course project for CSN-526: Machine Learning under [Professor Pravendra Singh](https://sites.google.com/view/pravendra/) at CSE Department, IIT Roorkee. The code is open-sourced under the MIT License.

### Team Members
