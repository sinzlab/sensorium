import numpy as np
import torch

from neuralpredictors.measures.np_functions import (
    corr,
    fev,
    oracle_corr_conservative,
    oracle_corr_jackknife,
    explainable_var,
    snr,
    anscombe,
)


def get_repeated_outputs(dataloader, min_repeats=2):
    # save the responses of all neuron to the repeats of an image as an element in a list
    repeated_inputs = []
    repeated_outputs = []
    for batch in dataloader:
        inputs, outputs = list(batch[:2])
        if len(inputs.shape) == 5:
            inputs = np.squeeze(inputs.cpu().numpy(), axis=0)
            outputs = np.squeeze(outputs.cpu().numpy(), axis=0)
        else:
            inputs = inputs.cpu().numpy()
            outputs = outputs.cpu().numpy()
        r, n = outputs.shape  # number of frame repeats, number of neurons
        if (
            r < min_repeats
        ):  # minimum number of frame repeats to be considered for oracle, free choice
            continue

        assert np.all(
            np.abs(np.diff(inputs[:, :1, ...], axis=0)) == 0
        ), "Images of oracle trials do not match"
        repeated_inputs.append(inputs)
        repeated_outputs.append(outputs)
    return np.array(repeated_inputs), np.array(repeated_outputs)


def get_jackknife_oracles(dataloaders, as_dict=False, per_neuron=True):
    oracles = {}
    for k, v in dataloaders.items():
        _, outputs = get_repeated_outputs(v)
        oracles[k] = oracle_corr_jackknife(np.array(outputs))
    if not as_dict:
        oracles = (
            np.hstack([v for v in oracles.values()])
            if per_neuron
            else np.mean(np.hstack([v for v in oracles.values()]))
        )
    return oracles


def get_conservative_oracles(dataloaders, as_dict=False, per_neuron=True):
    oracles = {}
    for k, v in dataloaders.items():
        _, outputs = get_repeated_outputs(v)
        oracles[k] = oracle_corr_conservative(np.array(outputs))
    if not as_dict:
        oracles = (
            np.hstack([v for v in oracles.values()])
            if per_neuron
            else np.mean(np.hstack([v for v in oracles.values()]))
        )
    return oracles


def get_explainable_var(
    dataloaders, as_dict=False, per_neuron=True, repeat_limit=None, randomize=True
):
    dataloaders = dataloaders["test"] if "test" in dataloaders else dataloaders
    exp_vars = {}
    for k, v in dataloaders.items():
        _, outputs = get_repeated_outputs(v)
        exp_vars[k] = compute_explainable_var(outputs)
    if not as_dict:
        exp_vars = (
            np.hstack([v for v in exp_vars.values()])
            if per_neuron
            else np.mean(np.hstack([v for v in exp_vars.values()]))
        )
    return exp_vars


def compute_explainable_var(outputs, eps=1e-9):
    ImgVariance = []
    TotalVar = np.var(np.vstack(outputs), axis=0, ddof=1)
    for out in outputs:
        ImgVariance.append(np.var(out, axis=0, ddof=1))
    ImgVariance = np.vstack(ImgVariance)
    NoiseVar = np.mean(ImgVariance, axis=0)
    explainable_var = (TotalVar - NoiseVar) / (TotalVar + eps)
    return explainable_var


def get_fano_factor(dataloaders, as_dict=False, per_neuron=True):
    """
    Returns average firing rate across the whole dataset
    """

    fano_factor = {}
    for k, dataloader in dataloaders.items():
        target = torch.empty(0)
        for batch in dataloader:
            images, responses = list(batch)[:2]
            if len(images.shape) == 5:
                responses = responses.squeeze(dim=0)
            target = torch.cat((target, responses.detach().cpu()), dim=0)
        fano_factor[k] = (target.var(0) / target.mean(0)).numpy()

    if not as_dict:
        fano_factor = (
            np.hstack([v for v in fano_factor.values()])
            if per_neuron
            else np.mean(np.hstack([v for v in fano_factor.values()]))
        )
    return fano_factor


def get_SNR(dataloaders, as_dict=False, per_neuron=True):
    SNRs = {}
    for k, dataloader in dataloaders.items():
        # assert isinstance(dataloader.batch_sampler, RepeatsBatchSampler), 'dataloader.batch_sampler must be a RepeatsBatchSampler'
        responses = []
        for batch in dataloader:
            images, resp = list(batch)[:2]
            responses.append(anscombe(resp.data.cpu().numpy()))
        mu = np.array([np.mean(repeats, axis=0) for repeats in responses])
        mu_bar = np.mean(mu, axis=0)
        sigma_2 = np.array([np.var(repeats, ddof=1, axis=0) for repeats in responses])
        sigma_2_bar = np.mean(sigma_2, axis=0)
        SNR = (1 / mu.shape[0] * np.sum((mu - mu_bar) ** 2, axis=0)) / sigma_2_bar
        SNRs[k] = SNR
    if not as_dict:
        SNRs = (
            np.hstack([v for v in SNRs.values()])
            if per_neuron
            else np.mean(np.hstack([v for v in SNRs.values()]))
        )
    return SNRs


def get_avg_firing(dataloaders, as_dict=False, per_neuron=True):
    """
    Returns average firing rate across the whole dataset
    """

    avg_firing = {}
    for k, dataloader in dataloaders.items():
        target = torch.empty(0)
        for batch in dataloader:
            images, responses = list(batch)[:2]
            if len(images.shape) == 5:
                responses = responses.squeeze(dim=0)
            target = torch.cat((target, responses.detach().cpu()), dim=0)
        avg_firing[k] = target.mean(0).numpy()

    if not as_dict:
        avg_firing = (
            np.hstack([v for v in avg_firing.values()])
            if per_neuron
            else np.mean(np.hstack([v for v in avg_firing.values()]))
        )
    return avg_firing
