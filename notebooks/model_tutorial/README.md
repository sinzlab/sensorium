# Neural Predictive Model Tutorial

Here, we provide a brief overview of our neural predictive model.

For our competition, we have trained two baselines:
- Our state-of-the-art CNN model
- a simple linear-nonlinear (LN) model

Have a look our demo notebook as an introduction as to how our [**CNN model**](./0_baseline_cnn.ipynb) works.

We also show how to train a model on multiple datasets at once, compute its performance, and show the retinotopy of the V1 neurons by relating the learned RF position to the anatomical coordinates with V1:
[**Generalization demo**](./2_model_evaluation_and_inspection.ipynb).

#### CPU vs GPU

We recommend to use a machine with a CUDA compatible GPU. On a CPU, the model training and evaluation is possible, too, but quite slow. To use CPU only, set `device=cpu` instead of `device=cuda` in the example notebooks.

# Video Explanation

- [**This Video**](https://youtu.be/xwLMO8nVvxs?t=220) goes into the details of how our model is built (prepared for ICLR 2021).



# References
- [Generalization in data-driven models of primary visual cortex.](https://www.biorxiv.org/content/10.1101/2020.10.05.326256v2)
  - the ICLR publication corresponding to the video above, which
- [Behavioral state tunes mouse vision to ethological features through pupil dilation](https://www.biorxiv.org/content/10.1101/2021.09.03.458870v2.full)
  - Uses a similar architecture as the CNN above, but also utilizes the behavioral variables to train the model. The same techniques are used in our baseline models for the **Sensorium+** track.