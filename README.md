<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![hub](https://img.shields.io/badge/powered%20by-hub%20-ff5a1f.svg)](https://github.com/activeloopai/Hub)

# SENSORIUM 2022 Competition

![plot](figures/Fig1.png)
SENSORIUM is a competition on predicting large scale mouse primary visual cortex activity. We will provide large scale datasets of neuronal activity in the visual cortex of mice. Participants will train models on pairs of natural stimuli and recorded neuronal responses, and submit the predicted responses to a set of test images for which responses are withheld.

Join our challenge and compete for the best neural predictive model!

For more information about the competition, vist our [website](https://sensorium2022.net/).

Have a look at our [White paper on arXiv](https://arxiv.org/abs/2206.08666), which describes the dataset and competition in detail.

# Important Dates
**June 20, 2022**: Start of the competition and data release.
<br>**Oct 15, 2022**: Submission deadline.
<br>**Oct 22, 2022**: Validation of all submitted scores completed. Preliminary winners are announced. Rank 1-3 in both competition tracks are contacted to provide the code for their submission.
<br>**Nov 5, 2022**: Deadline for top-ranked entries to provide the code for their submission.
<br>**Nov 15, 2022**: Winners contacted to contribute to the competition summary write-up.

# Starter-kit

Below we provide a step-by-step guide for getting started with the competition.

## 1. Pre-requisites
- install [**docker**](https://docs.docker.com/get-docker/) and [**docker-compose**](https://docs.docker.com/compose/install/)
- install git
- clone the repo via `git clone https://github.com/sinzlab/sensorium.git`

## 2. Download neural data

You can download the data from [https://gin.g-node.org/cajal/Sensorium2022](https://gin.g-node.org/cajal/Sensorium2022) and place it in `sensorium/notebooks/data`.
**Note:** Downloading the files all at once as a directory does lead to unfortunate errors. Thus, all datastes have to be downloaded individually.

## 3. Run the example notebooks

### **Start Jupyterlab environment**
```
cd sensorium/
docker-compose run -d -p 10101:8888 jupyterlab
```
now, type in `localhost:10101` in your favorite browser, and you are ready to go!


## **Competition example notebooks**
We provide notebooks that illustrate the structure of our data, our baselines models, and how to make a submission to the competition.
<br>[**Dataset tutorial**](notebooks/dataset_tutorial/): Shows the structure of the data and how to turn it into a PyTorch DataLoader.
<br>[**Model tutorial**](notebooks/model_tutorial/): How to train and evaluate our baseline models.
<br>[**Submission tutorial**](notebooks/submission_tutorial/): Use our API to make a submission to our competition.


If you have any questions, feel free to reach out to us (Contact section on our [website](https://sensorium2022.net/)), or raise an issue here on GitHub!
