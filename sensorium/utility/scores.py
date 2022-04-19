import types
import contextlib
import warnings
from itertools import combinations

import numpy as np
import torch

from .measure_helpers import get_subset_of_repeats, is_ensemble_function

from neuralpredictors.measures.np_functions import (
    corr,
    fev,
    oracle_corr_conservative,
    oracle_corr_jackknife,
    explainable_var,
    snr,
)
from neuralpredictors.training import eval_state, device_state


def model_predictions_repeats(
    model, dataloader, data_key, device="cuda", broadcast_to_target=False
):
    """
    Computes model predictions for a dataloader that yields batches with identical inputs along the first dimension.
    Unique inputs will be forwarded only once through the model
    Returns:
        target: ground truth, i.e. neuronal firing rates of the neurons as a list: [num_images][num_reaps, num_neurons]
        output: responses as predicted by the network for the unique images. If broadcast_to_target, returns repeated
                outputs of shape [num_images][num_reaps, num_neurons] else (default) returns unique outputs of shape [num_images, num_neurons]
    """
    target, output = [], []
    unique_images = torch.empty(0).to(device)
    for batch in dataloader:
        batch_args = list(batch)
        batch_kwargs = batch._asdict() if not isinstance(batch, dict) else batch
        images, responses = batch_args[:2]

        if len(images.shape) == 5:
            images = images.squeeze(dim=0)
            responses = responses.squeeze(dim=0)

        assert torch.all(
            torch.eq(
                images[-1, :1, ...],
                images[0, :1, ...],
            )
        ), "All images in the batch should be equal"
        unique_images = torch.cat(
            (
                unique_images,
                images[
                    0:1,
                ].to(device),
            ),
            dim=0,
        )
        target.append(responses.detach().cpu().numpy())

        if len(batch) > 2:
            with eval_state(model) if not is_ensemble_function(
                model
            ) else contextlib.nullcontext():
                with device_state(model, device) if not is_ensemble_function(
                    model
                ) else contextlib.nullcontext():
                    output.append(
                        model(*batch_args, data_key=data_key, **batch_kwargs)
                        .detach()
                        .cpu()
                        .numpy()
                    )

    # Forward unique images once
    with torch.no_grad():
        if len(output) == 0:
            with eval_state(model) if not is_ensemble_function(
                model
            ) else contextlib.nullcontext():
                with device_state(model, device) if not is_ensemble_function(
                    model
                ) else contextlib.nullcontext():
                    output = (
                        model(unique_images.to(device), data_key=data_key)
                        .detach()
                        .cpu()
                    )

                output = output.numpy()

    if broadcast_to_target:
        output = [np.broadcast_to(x, target[idx].shape) for idx, x in enumerate(output)]
    return target, output


def model_predictions(model, dataloader, data_key, device="cpu"):
    """
    computes model predictions for a given dataloader and a model
    Returns:
        target: ground truth, i.e. neuronal firing rates of the neurons
        output: responses as predicted by the network
    """

    target, output = torch.empty(0), torch.empty(0)
    for batch in dataloader:
        # batch_args = list(batch)
        # batch_kwargs = batch._asdict() if not isinstance(batch, dict) else batch
        images, responses = (
            batch[:2]
            if not isinstance(batch, dict)
            else (batch["inputs"], batch["targets"])
        )
        with torch.no_grad():
            with device_state(model, device) if not is_ensemble_function(
                model
            ) else contextlib.nullcontext():
                output = torch.cat(
                    (
                        output,
                        (model(images.to(device), data_key=data_key).detach().cpu()),
                    ),
                    dim=0,
                )
            target = torch.cat((target, responses.detach().cpu()), dim=0)

    return target.numpy(), output.numpy()


def get_signal_correlations(
    model, dataloaders, device="cpu", as_dict=False, per_neuron=True
):
    """
    Returns correlation between model outputs and average responses over repeated trials

    """
    if "test" in dataloaders:
        dataloaders = dataloaders["test"]

    correlations = {}
    for k, loader in dataloaders.items():

        # Compute correlation with average targets
        target, output = model_predictions_repeats(
            dataloader=loader,
            model=model,
            data_key=k,
            device=device,
            broadcast_to_target=False,
        )

        target_mean = np.array([t.mean(axis=0) for t in target])
        output_mean = (
            np.array([t.mean(axis=0) for t in output])
            if target[0].shape == output[0].shape
            else output
        )
        correlations[k] = corr(target_mean, output_mean, axis=0)

        # Check for nans
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


def get_correlations(
    model, dataloaders, device="cpu", as_dict=False, per_neuron=True, **kwargs
):
    correlations = {}
    with eval_state(model) if not is_ensemble_function(
        model
    ) else contextlib.nullcontext():
        for k, v in dataloaders.items():
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
    with torch.no_grad():
        with eval_state(model) if not is_ensemble_function(
            model
        ) else contextlib.nullcontext():
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


def get_fraction_oracles(model, dataloaders, device="cpu", conservative=False):
    dataloaders = dataloaders["test"] if "test" in dataloaders else dataloaders
    if conservative:
        oracles = get_get_oracles_correctedoracles_corrected(
            dataloaders=dataloaders, as_dict=False, per_neuron=True
        )
    else:
        oracles = get_oracles(dataloaders=dataloaders, as_dict=False, per_neuron=True)
    test_correlation = get_correlation(
        model=model,
        dataloaders=dataloaders,
        device=device,
        as_dict=False,
        per_neuron=True,
    )
    oracle_performance, _, _, _ = np.linalg.lstsq(
        np.hstack(oracles)[:, np.newaxis], np.hstack(test_correlation)
    )
    return oracle_performance[0]


def get_FEV(
    model,
    dataloaders,
    device="cpu",
    as_dict=False,
    per_neuron=True,
):
    """
    Computes the fraction of explainable variance explained (FEVe) per Neuron, given a model and a dictionary of dataloaders.
    The dataloaders will have to return batches of identical images, with the corresponing neuronal responses.

    Args:
        model (object): PyTorch module
        dataloaders (dict): Dictionary of dataloaders, with keys corresponding to "data_keys" in the model
        device (str): 'cuda' or 'gpu
        as_dict (bool): Returns the scores as a dictionary ('data_keys': values) if set to True.
        per_neuron (bool): Returns the grand average if set to True.
        threshold (float): for the avg feve, excludes neurons with a explainable variance below threshold

    Returns:
        FEV (dict, or np.array, or float): Fraction of explainable varianced explained. Per Neuron or as grand average.
    """
    dataloaders = dataloaders["test"] if "test" in dataloaders else dataloaders
    FEV = {}
    with eval_state(model) if not is_ensemble_function(
        model
    ) else contextlib.nullcontext():
        for data_key, dataloader in dataloaders.items():
            targets, outputs = model_predictions_repeats(
                model=model,
                dataloader=dataloader,
                data_key=data_key,
                device=device,
                broadcast_to_target=True,
            )
            FEV[data_key] = fev(targets, outputs)

    if not as_dict:
        FEV = (
            np.hstack([v for v in FEV.values()])
            if per_neuron
            else np.mean(np.hstack([v for v in FEV.values()]))
        )
    return FEV
