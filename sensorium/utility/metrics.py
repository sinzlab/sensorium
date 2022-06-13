import numpy as np
from neuralpredictors.measures.np_functions import corr, fev


class Metrics:
    def __init__(self, responses, trial_idx, image_ids, neuron_ids):
        """
        Computes performance metrics of neural response predictions.

        Args:
            responses (np.array): Recorded neural responses. Shape: (n_trials, n_neurons)
            trial_idx (np.array): trial indices of responses. Shape: (n_trials,)
            image_ids (np.array): image ids of responses. Shape: (n_trials,)
            neuron_ids (np.array): neuron ids of responses. Shape: (n_neurons,)
        """

        responses, trial_idx, image_ids, neuron_ids = self.order(
            responses, trial_idx, image_ids, neuron_ids
        )

        self.responses = responses
        self.trial_idx = trial_idx
        self.image_ids = image_ids
        self.neuron_ids = neuron_ids

    def order(self, responses, trial_idx, image_ids, neuron_ids):
        """
        Re-order the responses, ids, and indices based on ascending trial indices.

        Args:
            responses (np.array): Recorded neural responses. Shape: (n_trials, n_neurons)
            trial_idx (np.array): trial indices of responses. Shape: (n_trials,)
            image_ids (np.array): image ids of responses. Shape: (n_trials,)
            neuron_ids (np.array): neuron ids of responses. Shape: (n_neurons,)

        Returns:
            tuple: Re-ordered responses, trial_idx, image_ids, and neuron_ids
        """

        trial_idx_sorting_indices = np.argsort(trial_idx)
        neuron_ids_sorting_indices = np.argsort(neuron_ids)

        return (
            responses[trial_idx_sorting_indices, :][:, neuron_ids_sorting_indices],
            trial_idx[trial_idx_sorting_indices],
            image_ids[trial_idx_sorting_indices],
            neuron_ids[neuron_ids_sorting_indices],
        )

    def check_equality(self, trial_idx_submitted_ordered, neuron_ids_submitted_ordered):
        """
        Checks whether the (ordered) submitted and reference indices match.

        Args:
            trial_idx_submitted_ordered (np.array): ordered trial indices of predictions. Shape: (n_trials,)
            neuron_ids_submitted_ordered (np.array): ordered neuron ids of predictions. Shape: (n_neurons,)
        """

        assert np.equal(
            self.trial_idx, trial_idx_submitted_ordered
        ).all(), "trial_idx do not match"
        assert np.equal(
            self.neuron_ids, neuron_ids_submitted_ordered
        ).all(), "neuron_ids do not match"

    def split_images(self, responses):
        """
        Split the responses (or predictions) array based on image ids. Each element of the list contains
        the responses to repeated presentations of a single image.

        Args:
            responses (np.array): Recorded neural responses, or predictions. Shape: (n_trials, n_neurons)

        Returns:
            list: responses or predictios split across images. [n_images] np.array(n_repeats, n_neurons)
        """

        per_image_repeats = []
        for image_id in np.unique(self.image_ids):
            responses_across_repeats = responses[self.image_ids == image_id]
            per_image_repeats.append(responses_across_repeats)

        return per_image_repeats

    def correlation_to_single_trials(
        self,
        predictions_submitted,
        trial_idx_submitted,
        neuron_id_submitted,
        per_neuron=False,
    ):
        """
        Compute single-trial correlation.

        Args:
            predictions_submitted (np.array): Submitted predictions. Shape: (n_trials, n_neurons)
            trial_idx_submitted (np.array): Submitted trial indices. Shape: (n_trials,)
            neuron_id_submitted (np.array): Submitted neuron ids. Shape: (n_neurons,)
            per_neuron (bool): Whether to compute the measure per neuron.  Default is False.

        Returns:
            np.array or float: Correlation (single-trial) between responses and predictions
        """

        # get submitted stuff in the same order of the reference
        predictions, trial_idx, _, neuron_ids = self.order(
            predictions_submitted,
            trial_idx_submitted,
            np.zeros_like(trial_idx_submitted),
            neuron_id_submitted,
        )

        # check if trial indices and neuron ids are the same as the reference
        self.check_equality(trial_idx, neuron_ids)

        correlation = corr(predictions, self.responses, axis=0)
        return correlation if per_neuron else correlation.mean()

    def correlation_to_mean_across_repeats(
        self,
        predictions_submitted,
        trial_idx_submitted,
        neuron_id_submitted,
        per_neuron=False,
    ):
        """
        Compute correlation to average response across repeats.

        Args:
            predictions_submitted (np.array): Submitted predictions. Shape: (n_trials, n_neurons)
            trial_idx_submitted (np.array): Submitted trial indices. Shape: (n_trials,)
            neuron_id_submitted (np.array): Submitted neuron ids. Shape: (n_neurons,)
            per_neuron (bool): Whether to compute the measure per neuron.  Default is False.

        Returns:
            np.array or float: Correlation (average across repeats) between responses and predictions
        """

        # get submitted stuff in the same order of the reference
        predictions, trial_idx, _, neuron_ids = self.order(
            predictions_submitted,
            trial_idx_submitted,
            np.zeros_like(trial_idx_submitted),
            neuron_id_submitted,
        )

        # check if trial indices and neuron ids are the same as the reference
        self.check_equality(trial_idx, neuron_ids)

        # get repeats per image in a list
        mean_responses, mean_predictions = [], []
        for repeat_responses, repeat_predictions in zip(
            self.split_images(self.responses), self.split_images(predictions)
        ):
            mean_responses.append(repeat_responses.mean(axis=0, keepdims=True))
            mean_predictions.append(repeat_predictions.mean(axis=0, keepdims=True))

        mean_responses = np.vstack(mean_responses)
        mean_predictions = np.vstack(mean_predictions)

        correlation = corr(mean_responses, mean_predictions, axis=0)
        return correlation if per_neuron else correlation.mean()

    def feve(
        self,
        predictions_submitted,
        trial_idx_submitted,
        neuron_id_submitted,
        per_neuron=False,
        fev_threshold=0.15,
    ):
        """
        Compute fraction of explainable variance explained.

        Args:
            predictions_submitted (np.array): Submitted predictions. Shape: (n_trials, n_neurons)
            trial_idx_submitted (np.array): Submitted trial indices. Shape: (n_trials,)
            neuron_id_submitted (np.array): Submitted neuron ids. Shape: (n_neurons,)
            per_neuron (bool): Whether to compute the measure per neuron.  Default is False.

        Returns:
            np.array or float: FEVE by predictions
        """

        # get submitted stuff in the same order of the reference
        predictions, trial_idx, _, neuron_ids = self.order(
            predictions_submitted,
            trial_idx_submitted,
            np.zeros_like(trial_idx_submitted),
            neuron_id_submitted,
        )

        # check if trial indices and neuron ids are the same as the reference
        self.check_equality(trial_idx, neuron_ids)

        fev_val, feve_val = fev(
            self.split_images(self.responses),
            self.split_images(predictions),
            return_exp_var=True,
        )

        # ignore neurons below FEV threshold
        feve_val = feve_val[fev_val >= fev_threshold]

        return feve_val if per_neuron else feve_val.mean()
