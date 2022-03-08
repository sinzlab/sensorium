from eval import evaluate
import os
import click
import json
from pathlib import Path

FOLDER = "output"


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

