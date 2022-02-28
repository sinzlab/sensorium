import ast

import numpy as np
import pandas as pd

from nnfabrik.builder import get_data

from .utility.metrics import Metrics

ground_truth_data = ['static26645-2-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip',
                     'static26644-14-17-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip']


def load_submission_data(submission_path):
    """
    Extract necessary data for model evaluation from the submitted csv file.

    Args:
        submission_path (str): Complete path to the submission file.

    Returns:
        tuple: Contains:
               - trial indices (1D array)
               - image IDs (1D array)
               - neuron IDs (1D array)
               - predictions (2d array: trials x neurons)
    """
    submission_df = pd.read_csv(submission_path)
    trial_idx = submission_df["trial_indices"].values
    image_ids = submission_df["image_ids"].values
    neuron_ids = np.array(ast.literal_eval(submission_df["neuron_ids"].values[0]))
    predictions = np.array(
        [ast.literal_eval(v) for v in submission_df["prediction"].values]
    )

    return trial_idx, image_ids, neuron_ids, predictions


def load_groundtruth_data(benchmark=0, ):
    """
    Extract necessary data for model evaluation from the ground truth data file.

    Args:
        benchmark (int): Specifies which of benchmark to get the ground truth data from.
            0:  stimulus-responses challenge, data_key: 26645-2-18
            1:  brain state challenge, data_key: 26644-14-17

    Returns:
        tuple: Contains:
               - trial indices (1D array)
               - image IDs (1D array)
               - neuron IDs (1D array)
               - responses (2d array: trials x neurons)
    """


    dataset_fn = 'cascade.datasets.static_loaders'
    dataset_config = {'paths': [ground_truth_data[benchmark]],
                      'normalize': True,
                      'batch_size': 64,
                      }
    dataloaders = get_data(dataset_fn, dataset_config)
    data_key = list(dataloaders["test"].keys())[0]
    dat = dataloaders["train"][data_key].dataset

    neuron_ids = dat.neurons.unit_ids
    tiers = dat.trial_info.tiers
    complete_image_ids = dat.trial_info.frame_image_id
    complete_trial_idx = dat.trial_info.trial_idx

    trial_idx, responses, image_ids = [], [], []
    for i, datapoint in enumerate(dataloaders["train"][data_key].dataset):
        if tiers[i] != "test":
            continue

        trial_idx.append(complete_trial_idx[i])
        image_ids.append(complete_image_ids[i])
        responses.append(datapoint.responses.cpu().numpy().squeeze())

    trial_idx = np.array(trial_idx)
    image_ids = np.array(image_ids)
    responses = np.stack(responses)

    return trial_idx, image_ids, neuron_ids, responses



def evaluate(submission_path, ground_truth_path):
    """
    Compute evaluation metrics for a specific submission given the ground truth data.

    Args:
        submission_path (str): Absolute path to the submission csv file.
        ground_truth_path (str): Absolute path to the ground truth data file.

    Returns:
        dict: Containing all the evaluation results for all the evaluation metrics.
    """
    trial_idx_gt, image_ids_gt, neuron_ids_gt, responses = load_groundtruth_data(
        ground_truth_path
    )
    (
        trial_idx_submitted,
        image_ids_submitted,
        neuron_ids_submitted,
        predictions,
    ) = load_submission_data(submission_path)

    metric = Metrics(responses, trial_idx_gt, image_ids_gt, neuron_ids_gt)

    output = {}
    output["Correlation (single trial)"] = metric.correlation_to_single_trials(
        predictions, trial_idx_submitted, neuron_ids_submitted, per_neuron=False
    )
    output["Correlation (mean)"] = metric.correlation_to_mean_across_repeats(
        predictions, trial_idx_submitted, neuron_ids_submitted, per_neuron=False
    )
    output["FEVE"] = metric.feve(
        predictions, trial_idx_submitted, neuron_ids_submitted, per_neuron=False
    )
    return output
