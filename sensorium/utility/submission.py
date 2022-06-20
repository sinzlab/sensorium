import os
import pandas as pd
import torch
import numpy as np

from nnfabrik.builder import get_data
from neuralpredictors.training import eval_state, device_state
from neuralpredictors.data.datasets import FileTreeDataset


def model_predictions(model, dataloader, data_key, device="cpu"):
    """
    computes model predictions for a given dataloader and a model
    Returns:
        output: responses as predicted by the network
    """
    output = torch.empty(0)
    for batch in dataloader:
        images = batch[0] if not isinstance(batch, dict) else batch["inputs"]

        batch_kwargs = batch._asdict() if not isinstance(batch, dict) else batch
        batch_kwargs = {k: v.to(device) for k, v in batch_kwargs.items()}

        with torch.no_grad():
            with device_state(model, device):
                output = torch.cat(
                    (
                        output,
                        (model(images.to(device), data_key=data_key, **batch_kwargs).detach().cpu()),
                    ),
                    dim=0,
                )

    return output.numpy()


def get_data_filetree_loader(filename=None, dataloader=None, tier="test"):
    """
    Extracts necessary data for model evaluation from a dataloader based on the FileTree dataset.

    Args:
        filename (str): Specifies a path to the FileTree dataset.
        dataloader (obj): PyTorch Dataloader

    Returns:
        tuple: Contains:
               - trial indices (1D array)
               - image IDs (1D array)
               - neuron IDs (1D array)
               - responses (2d array: trials x neurons)
    """

    if dataloader is None:
        dataset_fn = "sensorium.datasets.static_loaders"
        dataset_config = {
            "paths": filename,
            "normalize": True,
            "batch_size": 64,
            "tier": tier,
        }
        dataloaders = get_data(dataset_fn, dataset_config)
        data_key = list(dataloaders[tier].keys())[0]

        dat = dataloaders[tier][data_key].dataset
    else:
        dat = dataloader.dataset

    neuron_ids = dat.neurons.unit_ids.tolist()
    tiers = dat.trial_info.tiers
    complete_image_ids = dat.trial_info.frame_image_id
    complete_trial_idx = dat.trial_info.trial_idx

    trial_indices, responses, image_ids = [], [], []
    for i, datapoint in enumerate(dat):
        if tiers[i] != tier:
            continue

        trial_indices.append(complete_trial_idx[i])
        image_ids.append(complete_image_ids[i])
        responses.append(datapoint.responses.cpu().numpy().squeeze())

    responses = np.stack(responses)

    return trial_indices, image_ids, neuron_ids, responses


def get_data_hub_loader(dataloader):
    """
    Extracts necessary data for model evaluation from a dataloader based on hub.

    Args:
        dataloader (obj): PyTorch Dataloader

    Returns:
        tuple: Contains:
               - trial indices (1D array)
               - image IDs (1D array)
               - neuron IDs (1D array)
    """
    image_ids = dataloader.dataset.dataset.image_ids.data().flatten().tolist()
    trial_indices = dataloader.dataset.dataset.trial_indices.data().flatten().tolist()
    neuron_ids = dataloader.dataset.dataset.info["neuron_ids"]
    return trial_indices, image_ids, neuron_ids


def generate_submission_file(
    trained_model, dataloaders, data_key=None, path=None, device="cpu", tier=None,
):
    """
    Helper function to create the submission .csv file, given a trained model and the dataloader.

    Args:
        trained_model (nn.module): model trained on the respective benchmark data.
        dataloader (dict): dataloader from the respective benchmark data, has to contain the
                                 "test" and "final_test" keys for the competition.
        data_key (str, optional): specifies the data_key, if the model was trained on many datasets
        path (str, optional): output directory of the .csv file
        device (str): device name to which model and input images are cast to.

    Returns:
        None. the output .csv file will be saved in the specified path, or relative to the user's current working directory.
    """
    if tier is None:
        tier_list = ["test", "final_test"]
    else:
        tier_list = [tier]

    for tier in tier_list:
        test_dataloader = dataloaders[tier][data_key]

        test_predictions = model_predictions(
            trained_model,
            test_dataloader,
            data_key=data_key,
            device=device,
        )

        if isinstance(test_dataloader.dataset, FileTreeDataset):
            trial_indices, image_ids, neuron_ids, _ = get_data_filetree_loader(
                dataloader=test_dataloader, tier=tier,
            )
        else:
            trial_indices, image_ids, neuron_ids = get_data_hub_loader(
                dataloader=test_dataloader, tier=tier,
            )

        df = pd.DataFrame(
            {
                "trial_indices": trial_indices,
                "image_ids": image_ids,
                "prediction": test_predictions.tolist(),
                "neuron_ids": [neuron_ids] * len(test_predictions),
            }
        )
        tier_name = tier if tier != "test" else "live_test"
        submission_filename = f"submission_file_{tier_name}.csv"
        save_path = os.path.join(path, submission_filename) if path is not None else submission_filename
        df.to_csv(save_path, index=False)
        print(f"Submission file saved for tier: {tier_name}. Saved in: {save_path}")


def generate_ground_truth_file(
    filename,
    path=None,
    tier=None,
):
    """
    Extract necessary data for model evaluation from the ground truth data file.

    Args:
        filename (str): Specifies which of benchmark datasets to get the ground truth data from.
        path (str): output directory, where the files are getting created
        tier (str): the tier, for which the ground truth file is getting creates.
                    By default, creates two files, for "final_test" and "test" tiers of the SENSORIUM
                    and SENSORIUM+ competition datasets.
                    But it can also be used to generate a ground truth file for one of the "pre-training" scans.
                        In this case, tier has to be set to "test".

    Returns:
        saves one or two output .csv files at the specified path.
    """
    if tier is None:
        tier_list = ["test", "final_test"]
    else:
        tier_list = [tier]

    for tier in tier_list:
        trial_indices, image_ids, neuron_ids, responses = get_data_filetree_loader(
            filename=filename,
            tier=tier,
        )
        df = pd.DataFrame(
            {
                "trial_indices": trial_indices,
                "image_ids": image_ids,
                "responses": responses.tolist(),
                "neuron_ids": [neuron_ids] * len(responses),
            }
        )
        gt_filename = f"ground_truth_file_{tier}.csv"
        save_path = os.path.join(path, gt_filename) if path is not None else gt_filename
        df.to_csv(save_path, index=False)
        print(f"Submission file saved for tier: {tier}. Saved in: {save_path}")
