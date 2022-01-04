from nnfabrik.templates.trained_model import TrainedModelBase

from nnfabrik.main import my_nnfabrik

main_nnfabrik = my_nnfabrik(
    schema="nnfabrik_neural_prediction_challenge",
    use_common_fabrikant=False,
)

Fabrikant, Dataset, Model, Trainer, Seed = map(
    main_nnfabrik.__dict__.get, ["Fabrikant", "Dataset", "Model", "Trainer", "Seed"]
)


@main_nnfabrik.schema
class TrainedModel(TrainedModelBase):
    table_comment = "Trained models"
    nnfabrik = main_nnfabrik
