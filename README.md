# Starter-Kit for the SENSORIUM 2022 competition
Code base for The SENSORIUM competition on predicting large scale mouse primary visual cortex activity.

For more information about the competition, vist our [website](http://www.sensorium2022.net/)!

Competition Data: Will be released on June 1st, 2022

Demo Data to get familiar with the dataset structure: https://gin.g-node.org/cajal/Lurz2020

## The Competition
![plot](./Fig1.png)
An illustration of the SENSORIUM competition. We will provide large scale
datasets of neuronal activity in the visual cortex of mice. Participants will train models on pairs
of natural images stimuli and recorded neuronal responses, and submit the predicted responses to
a set of test images for which responses are withheld.

Join our challenge and compete for the best neural predictive model!

# Installation
## Requirements
* `docker` and `docker-compose`

## Quickstart

Navigate to a folder of your choice and run the following commands in a [shell of your choice](https://fishshell.com/):

```bash
# clone this repo
git clone https://github.com/sinzlab/sensorium.git

# get the data

# Option 1: login in via gin
cd Lurz_2020_code/notebooks/data
gin login
gin get cajal/Lurz2020 # might take a while; fast internet recommended

# Option 2:
# download the data from https://gin.g-node.org/cajal/Lurz2020
# unzip it into sensorium/notebooks/data

# create docker container 
cd sensorium/
docker-compose run notebook
```

# Example Notebooks
We provide four notebooks that illustrate the structure of our data, our baselines models, and how to make a submission to the competition.
### Notebook 1: Inspecting the Data
### Notebook 2: Re-train our Baseline Models
### Notebook 3: Use our API to make a submission to our competition
### Notebook 4: A Full submission in 5 easy steps using our cloud-based DataLoaders
