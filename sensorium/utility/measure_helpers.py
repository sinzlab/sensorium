import types
import warnings
import numpy as np
import math


def get_subset_of_repeats(outputs, repeat_limit, randomize=True):
    """
    Args:
        outputs (array or list): repeated responses/targets to the same input. with the shape [inputs, ] [reps, neurons]
                                    or array(inputs, reps, neurons)
        repeat_limit (int): how many reps are selected
        randomize (cool): if True, takes a random selection of repetitions. if false, takes the first n repetitions.

    Returns: limited_outputs (list): same shape as inputs, but with reduced number of repetitions

    """
    limited_output = []
    for repetitions in outputs:
        n_repeats = repetitions.shape[0]
        limited_output.append(
            repetitions[
                :repeat_limit,
            ]
            if not randomize
            else repetitions[
                np.random.choice(
                    n_repeats,
                    repeat_limit if repeat_limit < n_repeats else n_repeats,
                    replace=False,
                )
            ]
        )
    return limited_output


def is_ensemble_function(model):
    return isinstance(model, types.FunctionType)
