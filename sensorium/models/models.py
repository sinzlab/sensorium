from torch import nn

from nnfabrik.utility.nn_helpers import set_random_seed, get_dims_for_loader_dict
from neuralpredictors.utils import get_module_output
from neuralpredictors.layers.encoders import FiringRateEncoder
from neuralpredictors.layers.shifters import MLPShifter, StaticAffine2dShifter
from neuralpredictors.layers.cores import (
    Stacked2dCore,
    SE2dCore,
    RotationEquivariant2dCore,
)

from .readouts import MultipleFullGaussian2d
from .utility import prepare_grid


def stacked_core_full_gauss_readout(
    dataloaders,
    seed,
    hidden_channels=32,
    input_kern=13,
    hidden_kern=3,
    layers=3,
    gamma_input=15.5,
    skip=0,
    final_nonlinearity=True,
    momentum=0.9,
    pad_input=False,
    batch_norm=True,
    hidden_dilation=1,
    laplace_padding=None,
    input_regularizer="LaplaceL2norm",
    use_avg_reg=False,
    init_mu_range=0.2,
    init_sigma=1.0,
    readout_bias=True,
    gamma_readout=4,
    elu_offset=0,
    stack=None,
    depth_separable=False,
    linear=False,
    gauss_type="full",
    grid_mean_predictor=None,
    attention_conv=False,
    shifter=None,
    shifter_type="MLP",
    input_channels_shifter=2,
    hidden_channels_shifter=5,
    shift_layers=3,
    gamma_shifter=0,
    shifter_bias=True,
    hidden_padding=None,
    core_bias=True,
):
    """
    Model class of a stacked2dCore (from neuralpredictors) and a pointpooled (spatial transformer) readout

    Args:
        dataloaders: a dictionary of dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: random seed
        grid_mean_predictor: if not None, needs to be a dictionary of the form
            {
            'type': 'cortex',
            'input_dimensions': 2,
            'hidden_layers':0,
            'hidden_features':20,
            'final_tanh': False,
            }
            In that case the datasets need to have the property `neurons.cell_motor_coordinates`
        share_features: whether to share features between readouts. This requires that the datasets
            have the properties `neurons.multi_match_id` which are used for matching. Every dataset
            has to have all these ids and cannot have any more.
        all other args: See Documentation of Stacked2dCore in neuralpredictors.layers.cores and
            PointPooled2D in neuralpredictors.layers.readouts

    Returns: An initialized model which consists of model.core and model.readout
    """

    if "train" in dataloaders.keys():
        dataloaders = dataloaders["train"]

    # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
    batch = next(iter(list(dataloaders.values())[0]))
    in_name, out_name = (
        list(batch.keys())[:2] if isinstance(batch, dict) else batch._fields[:2]
    )

    session_shape_dict = get_dims_for_loader_dict(dataloaders)
    n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
    input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = (
        list(input_channels.values())[0]
        if isinstance(input_channels, dict)
        else input_channels[0]
    )

    set_random_seed(seed)
    grid_mean_predictor, grid_mean_predictor_type, source_grids = prepare_grid(grid_mean_predictor, dataloaders)

    core = Stacked2dCore(
        input_channels=core_input_channels,
        hidden_channels=hidden_channels,
        input_kern=input_kern,
        hidden_kern=hidden_kern,
        layers=layers,
        gamma_input=gamma_input,
        skip=skip,
        final_nonlinearity=final_nonlinearity,
        bias=core_bias,
        momentum=momentum,
        pad_input=pad_input,
        batch_norm=batch_norm,
        hidden_dilation=hidden_dilation,
        laplace_padding=laplace_padding,
        input_regularizer=input_regularizer,
        stack=stack,
        depth_separable=depth_separable,
        linear=linear,
        attention_conv=attention_conv,
        hidden_padding=hidden_padding,
        use_avg_reg=use_avg_reg,
    )

    in_shapes_dict = {
        k: get_module_output(core, v[in_name])[1:]
        for k, v in session_shape_dict.items()
    }

    readout = MultipleFullGaussian2d(
        in_shape_dict=in_shapes_dict,
        loader=dataloaders,
        n_neurons_dict=n_neurons_dict,
        init_mu_range=init_mu_range,
        bias=readout_bias,
        init_sigma=init_sigma,
        gamma_readout=gamma_readout,
        gauss_type=gauss_type,
        grid_mean_predictor=grid_mean_predictor,
        grid_mean_predictor_type=grid_mean_predictor_type,
        source_grids=source_grids,
    )

    if shifter is True:
        data_keys = [i for i in dataloaders.keys()]
        if shifter_type == "MLP":
            shifter = MLPShifter(
                data_keys=data_keys,
                input_channels=input_channels_shifter,
                hidden_channels_shifter=hidden_channels_shifter,
                shift_layers=shift_layers,
                gamma_shifter=gamma_shifter,
            )

        elif shifter_type == "StaticAffine":
            shifter = StaticAffine2dShifter(
                data_keys=data_keys,
                input_channels=input_channels_shifter,
                bias=shifter_bias,
                gamma_shifter=gamma_shifter,
            )

    model = FiringRateEncoder(
        core=core,
        readout=readout,
        shifter=shifter,
        elu_offset=elu_offset,
    )

    return model
