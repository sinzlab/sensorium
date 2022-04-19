from torch.utils.data import DataLoader
import numpy as np
from neuralpredictors.data.samplers import RepeatsBatchSampler


def get_oracle_dataloader(
    dat,
    toy_data=False,
    image_condition=None,
    verbose=False,
    file_tree=False,
    data_key=None,
    trial_idx_selection=None,
):

    if toy_data:
        condition_hashes = dat.info.condition_hash
    else:
        dat_info = dat.info if not file_tree else dat.trial_info
        if "image_id" in dir(dat_info):
            condition_hashes = dat_info.image_id
            image_class = dat_info.image_class

        elif "colorframeprojector_image_id" in dir(dat_info):
            condition_hashes = dat_info.colorframeprojector_image_id
            image_class = dat_info.colorframeprojector_image_class
        elif "frame_image_id" in dir(dat_info):
            condition_hashes = dat_info.frame_image_id
            image_class = dat_info.frame_image_class
        else:
            raise ValueError(
                "'image_id' 'colorframeprojector_image_id', or 'frame_image_id' have to present in the dataset under dat.info "
                "in order to load get the oracle repeats."
            )

    max_idx = condition_hashes.max() + 1
    classes, class_idx = np.unique(image_class, return_inverse=True)
    identifiers = condition_hashes + class_idx * max_idx
    dat_tiers = dat.tiers if not file_tree else dat.trial_info.tiers
    trial_idx = dat.trial_idx if not file_tree else dat.trial_info.trial_idx
    test_trial_selection = (
        (dat_tiers == "test")
        if trial_idx_selection is None
        else ((dat_tiers == "test") & np.isin(trial_idx, trial_idx_selection))
    )

    if image_condition is None:
        sampling_condition = np.where(test_trial_selection)[0]
    elif isinstance(image_condition, str):
        image_condition_filter = image_class == image_condition
        sampling_condition = np.where(
            (test_trial_selection) & (image_condition_filter)
        )[0]
    elif isinstance(image_condition, list):
        image_condition_filter = sum(
            [image_class == i for i in image_condition]
        ).astype(np.bool)
        sampling_condition = np.where(
            (test_trial_selection) & (image_condition_filter)
        )[0]
    else:
        raise TypeError(
            "image_condition argument has to be a string or list of strings"
        )

    if (image_condition is not None) and verbose:
        print(f"Created Testloader for image class {image_condition}")

    sampler = RepeatsBatchSampler(identifiers, sampling_condition)
    dataloaders = {}
    dataloaders[data_key] = DataLoader(dat, batch_sampler=sampler)
    return dataloaders


def get_validation_split(n_images, train_frac, seed):
    """
    Splits the total number of images into train and test set.
    This ensures that in every session, the same train and validation images are being used.

    Args:
        n_images: Total number of images. These will be plit into train and validation set
        train_frac: fraction of images used for the training set
        seed: random seed

    Returns: Two arrays, containing image IDs of the whole imageset, split into train and validation

    """
    if seed:
        np.random.seed(seed)
    train_idx, val_idx = np.split(
        np.random.permutation(int(n_images)), [int(n_images * train_frac)]
    )
    assert not np.any(
        np.isin(train_idx, val_idx)
    ), "train_set and val_set are overlapping sets"

    return train_idx, val_idx
