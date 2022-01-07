import torch


class Permute:
    def __init__(self, *permutation):
        """
        Applies permuation on the input dimensions. Assumes the input has
        3 dimensions (e.g. height, width, channels). If nothing is passed it assumes
        the input dimensions are in the above-mentioned order and permutes the
        dimensions as follows: channels, height, width.
        """
        if permutation:
            if len(permutation) != 3:
                raise ValueError
            else:
                self.permutation = permutation
        if not permutation:
            self.permutation = (2, 0, 1)

    def __call__(self, images):
        print(images.transpose(*self.permutation).shape)
