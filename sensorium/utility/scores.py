import warnings
import numpy as np
import torch

from neuralpredictors.measures.np_functions import corr, fev
from neuralpredictors.training import eval_state, device_state

from .submission import get_data_filetree_loader


def split_images(responses, image_ids):
    """
    Split the responses (or predictions) array based on image ids. Each element of the list contains
    the responses to repeated presentations of a single image.

    Args:
        responses (np.array): Recorded neural responses, or predictions. Shape: (n_trials, n_neurons)

    Returns:
        list: responses or predictios split across images. [n_images] np.array(n_repeats, n_neurons)
    """

    per_image_repeats = []
    for image_id in np.unique(image_ids):
        responses_across_repeats = responses[image_ids == image_id]
        per_image_repeats.append(responses_across_repeats)

    return per_image_repeats


def model_predictions(model, dataloader, data_key, device="cpu"):
    """
    computes model predictions for a given dataloader and a model
    Returns:
        target: ground truth, i.e. neuronal firing rates of the neurons
        output: responses as predicted by the network
    """

    target, output = torch.empty(0), torch.empty(0)
    for batch in dataloader:
        images, responses = (
            batch[:2]
            if not isinstance(batch, dict)
            else (batch["inputs"], batch["targets"])
        )
        batch_kwargs = batch._asdict() if not isinstance(batch, dict) else batch

        with torch.no_grad():
            with device_state(model, device):
                output = torch.cat(
                    (
                        output,
                        (
                            model(images.to(device), data_key=data_key, **batch_kwargs)
                            .detach()
                            .cpu()
                        ),
                    ),
                    dim=0,
                )
            target = torch.cat((target, responses.detach().cpu()), dim=0)

    return target.numpy(), output.numpy()


def get_correlations(
    model, dataloaders, tier=None, device="cpu", as_dict=False, per_neuron=True, **kwargs
):
    """
    Computes single-trial correlation between model prediction and true responses

    Args:
        model (torch.nn.Module): Model used to predict responses.
        dataloaders (dict): dict of test set torch dataloaders.
        tier(str): the data-tier (train/test/val). If tier is None, then it is assumed that the the tier-key is not present.
        device (str, optional): device to compute on. Defaults to "cpu".
        as_dict (bool, optional): whether to return the results per data_key. Defaults to False.
        per_neuron (bool, optional): whether to return the results per neuron or averaged across neurons. Defaults to True.

    Returns:
        dict or np.ndarray: contains the correlation values.
    """
    correlations = {}
    dl = dataloaders[tier] if tier is not None else dataloaders
    for k, v in dl.items():
        target, output = model_predictions(
            dataloader=v, model=model, data_key=k, device=device
        )
        correlations[k] = corr(target, output, axis=0)

        if np.any(np.isnan(correlations[k])):
            warnings.warn(
                "{}% NaNs , NaNs will be set to Zero.".format(
                    np.isnan(correlations[k]).mean() * 100
                )
            )
        correlations[k][np.isnan(correlations[k])] = 0

    if not as_dict:
        correlations = (
            np.hstack([v for v in correlations.values()])
            if per_neuron
            else np.mean(np.hstack([v for v in correlations.values()]))
        )
    return correlations


def get_signal_correlations(
    model, dataloaders, tier, device="cpu", as_dict=False, per_neuron=True
):
    """
    Same as `get_correlations` but first responses and predictions are averaged across repeats
    and then the correlation is computed. In other words, the correlation is computed between
    the means across repeats.
    """
    correlations = {}
    for data_key, dataloader in dataloaders[tier].items():
        trial_indices, image_ids, neuron_ids, responses = get_data_filetree_loader(
            dataloader=dataloader, tier=tier
        )
        _, predictions = model_predictions(
            model, dataloader, data_key=data_key, device=device
        )

        repeats_responses = split_images(responses, image_ids)
        repeats_predictions = split_images(predictions, image_ids)

        mean_responses, mean_predictions = [], []
        for repeat_responses, repeat_predictions in zip(
            repeats_responses, repeats_predictions
        ):
            mean_responses.append(repeat_responses.mean(axis=0, keepdims=True))
            mean_predictions.append(repeat_predictions.mean(axis=0, keepdims=True))

        mean_responses = np.vstack(mean_responses)
        mean_predictions = np.vstack(mean_predictions)

        correlations[data_key] = corr(mean_responses, mean_predictions, axis=0)

    if not as_dict:
        correlations = (
            np.hstack([v for v in correlations.values()])
            if per_neuron
            else np.mean(np.hstack([v for v in correlations.values()]))
        )

    return correlations if per_neuron else correlations.mean()


def get_fev(model, dataloaders, tier, device="cpu", per_neuron=True, fev_threshold=0.15, as_dict=False):
    """
    Compute the fraction of explainable variance explained per neuron.

    Args:
        model (torch.nn.Module): Model used to predict responses.
        dataloaders (dict): dict of test set torch dataloaders.
        tier (str): specify the tier for which fev should be computed.
        device (str, optional): device to compute on. Defaults to "cpu".
        per_neuron (bool, optional): whether to return the results per neuron or averaged across neurons. Defaults to True.
        fev_threshold (float): the FEV threshold under which a neuron will not be ignored.

    Returns:
        np.ndarray: the fraction of explainable variance explained.
    """
    feves = {}
    for data_key, dataloader in dataloaders[tier].items():
        trial_indices, image_ids, neuron_ids, responses = get_data_filetree_loader(
            dataloader=dataloader, tier=tier
        )
        _, predictions = model_predictions(
            model, dataloader, data_key=data_key, device=device
        )
        fev_val, feve_val = fev(
            split_images(responses, image_ids),
            split_images(predictions, image_ids),
            return_exp_var=True,
        )

        # ignore neurons below FEV threshold
        feve_val = feve_val[fev_val >= fev_threshold]

        feves[data_key] = feve_val

    if not as_dict:
        feves = (
            np.hstack([v for v in feves.values()])
            if per_neuron
            else np.mean(np.hstack([v for v in feves.values()]))
        )

    return feves if per_neuron else feves.mean()


def get_poisson_loss(
    model,
    dataloaders,
    device="cpu",
    as_dict=False,
    avg=False,
    per_neuron=True,
    eps=1e-12,
):
    poisson_loss = {}
    for k, v in dataloaders.items():
        target, output = model_predictions(
            dataloader=v, model=model, data_key=k, device=device
        )
        loss = output - target * np.log(output + eps)
        poisson_loss[k] = np.mean(loss, axis=0) if avg else np.sum(loss, axis=0)
    if as_dict:
        return poisson_loss
    else:
        if per_neuron:
            return np.hstack([v for v in poisson_loss.values()])
        else:
            return (
                np.mean(np.hstack([v for v in poisson_loss.values()]))
                if avg
                else np.sum(np.hstack([v for v in poisson_loss.values()]))
            )
