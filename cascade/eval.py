import os
import click
import json
import ast
import numpy as np
import pandas as pd
from cascade.utility.metrics import Metrics

FOLDER = "output"


def load_submission_data(submission_path):
    submission_df = pd.read_csv(submission_path)
    trial_idx = submission_df["trial_indices"].values
    image_ids = submission_df["image_ids"].values
    neuron_ids = np.array(ast.literal_eval(submission_df["neuron_ids"].values[0]))
    predictions = np.array(
        [ast.literal_eval(v) for v in submission_df["prediction"].values]
    )

    return trial_idx, image_ids, neuron_ids, predictions


def load_groundtruth_data(submission_path):
    submission_df = pd.read_csv(submission_path)
    trial_idx = submission_df["trial_indices"].values
    image_ids = submission_df["image_ids"].values
    neuron_ids = np.array(ast.literal_eval(submission_df["neuron_ids"].values[0]))
    predictions = np.array(
        [ast.literal_eval(v) for v in submission_df["prediction"].values]
    )

    return trial_idx, image_ids, neuron_ids, predictions


def evaluate(submission_path, ground_truth_path):
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
    output["correlation_to_single_trials"] = metric.correlation_to_single_trials(
        predictions, trial_idx_submitted, neuron_ids_submitted, per_neuron=False
    )
    output[
        "correlation_to_mean_across_repeats"
    ] = metric.correlation_to_mean_across_repeats(
        predictions, trial_idx_submitted, neuron_ids_submitted, per_neuron=False
    )
    output["feve"] = metric.feve(
        predictions, trial_idx_submitted, neuron_ids_submitted, per_neuron=False
    )
    return output


@click.command()
@click.option(
    "--submission_path",
    prompt="The submission_path",
    help="Provide the path where the submission is placed",
)
@click.option(
    "--ground_truth_path",
    prompt="Provide the path where the ground_truth is placed",
    help="The ground_truth_path",
)
@click.option(
    "--submission_id",
    prompt="Provide submission id",
    help="The submission_id to create a result",
)
def main(submission_path, ground_truth_path, submission_id):
    """ Templated execution file"""
    try:
        output = evaluate(submission_path, ground_truth_path)
        print(f"The evaluation output:{output}")
        if not os.path.exists(FOLDER):
            os.makedirs(FOLDER)
        with open(os.path.join(FOLDER, submission_id + ".json"), "w") as f:
            json.dump(output, f)
        print("Success:Completed the evaluation written to output dir")
    except Exception as e:
        print(e)
        print("Exception:There has been an error while executing file")


if __name__ == "__main__":
    main()
