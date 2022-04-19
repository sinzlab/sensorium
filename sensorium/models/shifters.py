from torch import nn
from torch.nn.init import xavier_normal
import torch
from torch.nn import ModuleDict


class Shifter:
    def __repr__(self):
        s = super().__repr__()
        s += " [{} regularizers: ".format(self.__class__.__name__)
        ret = []
        for attr in filter(lambda x: "gamma" in x, dir(self)):
            ret.append("{} = {}".format(attr, getattr(self, attr)))
        return s + "|".join(ret) + "]\n"


class MLP(nn.Module):
    def __init__(self, input_features=2, hidden_channels=10, shift_layers=1, **kwargs):
        super().__init__()

        feat = []
        if shift_layers > 1:
            feat = [nn.Linear(input_features, hidden_channels), nn.Tanh()]
        else:
            hidden_channels = input_features

        for _ in range(shift_layers - 2):
            feat.extend([nn.Linear(hidden_channels, hidden_channels), nn.Tanh()])

        feat.extend([nn.Linear(hidden_channels, 2), nn.Tanh()])
        self.mlp = nn.Sequential(*feat)

    def regularizer(self):
        return 0

    def initialize(self):
        for linear_layer in [p for p in self.parameters() if isinstance(p, nn.Linear)]:
            xavier_normal(linear_layer.weight)

    def forward(self, input):
        return self.mlp(input)


class MLPShifter(Shifter, ModuleDict):
    def __init__(
        self,
        data_keys,
        input_channels=2,
        hidden_channels_shifter=2,
        shift_layers=1,
        gamma_shifter=0,
        **kwargs
    ):
        super().__init__()
        self.gamma_shifter = gamma_shifter
        for k in data_keys:
            self.add_module(
                k, MLP(input_channels, hidden_channels_shifter, shift_layers)
            )

    def initialize(self, **kwargs):
        log.info(
            "Ignoring input {} when initializing {}".format(
                repr(kwargs), self.__class__.__name__
            )
        )
        for linear_layer in [p for p in self.parameters() if isinstance(p, nn.Linear)]:
            xavier_normal(linear_layer.weight)

    def regularizer(self, data_key):
        return self[data_key].regularizer() * self.gamma_shifter


class StaticAffine2dShifter(Shifter, ModuleDict):
    def __init__(self, data_keys, input_channels, bias=True, gamma_shifter=0, **kwargs):
        super().__init__()
        self.gamma_shifter = gamma_shifter
        for k in data_keys:
            self.add_module(k, StaticAffine2d(input_channels, 2, bias=bias))

    def initialize(self, bias=None):
        for k in self:
            if bias is not None:
                self[k].initialize(bias=bias[k])
            else:
                self[k].initialize()

    def regularizer(self, data_key):
        return self[data_key].weight.pow(2).mean() * self.gamma_shifter


class StaticAffine2d(nn.Linear):
    def __init__(self, input_channels, output_channels, bias=True, **kwargs):
        super().__init__(input_channels, output_channels, bias=bias)

    def forward(self, x):
        x = super().forward(x)
        return torch.tanh(x)

    def initialize(self, bias=None):
        self.weight.data.normal_(0, 1e-6)
        if self.bias is not None:
            if bias is not None:
                log.info("Setting bias to predefined value")
                self.bias.data = bias
            else:
                self.bias.data.normal_(0, 1e-6)


def NoShifter(*args, **kwargs):
    """
    Dummy function to create an object that returns None
    Args:
        *args:   will be ignored
        *kwargs: will be ignored

    Returns:
        None
    """
    return None
