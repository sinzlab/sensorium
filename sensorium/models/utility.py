import copy

#TODO: add docstring, add cell motor coord function
def prepare_grid(grid_mean_predictor, dataloaders):
    """
    Args:
        grid_mean_predictor:
        dataloaders:
    Returns:

    """
    if grid_mean_predictor is None:
        grid_mean_predictor_type = None
        source_grids = None
    else:
        grid_mean_predictor = copy.deepcopy(grid_mean_predictor)
        grid_mean_predictor_type = grid_mean_predictor.pop("type")

        if grid_mean_predictor_type == "cortex":
            input_dim = grid_mean_predictor.pop("input_dimensions", 2)
            source_grids = {
                k: v.dataset.neurons.cell_motor_coordinates[:, :input_dim]
                for k, v in dataloaders.items()
            }
    return grid_mean_predictor, grid_mean_predictor_type, source_grids