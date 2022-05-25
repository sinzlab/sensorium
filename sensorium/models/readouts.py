from neuralpredictors.layers.readouts import MultiReadoutSharedParametersBase, FullGaussian2d


class MultipleFullGaussian2d(MultiReadoutSharedParametersBase):
    _base_readout = FullGaussian2d
