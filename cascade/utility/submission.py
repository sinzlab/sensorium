import pandas as pd
import torch
from neuralpredictors.training import eval_state, device_state

# from cascade.utility.scores import model_predictions # why does this return targets?
def model_predictions(model, dataloader, data_key, device="cpu"):
    """
    computes model predictions for a given dataloader and a model
    Returns:
        output: responses as predicted by the network
    """
    output = torch.empty(0)
    for batch in dataloader:
        images = batch[0] if not isinstance(batch, dict) else batch["inputs"]
        with torch.no_grad():
            with device_state(model, device):
                output = torch.cat(
                    (
                        output,
                        (model(images.to(device), data_key=data_key).detach().cpu()),
                    ),
                    dim=0,
                )

    return output.numpy()


def generate_submission_file(trained_model, test_dataloader, data_key=None, path=None):
    test_predictions = model_predictions(
        trained_model, test_dataloader, data_key=data_key
    )
    image_ids = test_dataloader.dataset.dataset.image_ids.data().flatten().tolist()
    trial_indices = (
        test_dataloader.dataset.dataset.trial_indices.data().flatten().tolist()
    )
    neuron_ids = test_dataloader.dataset.dataset.info["neuron_ids"]

    df = pd.DataFrame(
        {
            "trial_indices": trial_indices,
            "image_ids": image_ids,
            "prediction": test_predictions.tolist(),
            "neuron_ids": [neuron_ids] * len(test_predictions),
        }
    )

    path = path if path is not None else "submission_file.csv"
    df.to_csv(path, index=False)

    print("File saved.")