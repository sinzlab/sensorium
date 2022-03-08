from neuralpredictors.layers.readouts import MultiReadoutBase, FullGaussian2d


class MultipleFullGaussian2d(MultiReadoutBase):
    _base_readout = FullGaussian2d
