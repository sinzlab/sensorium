import pandas as pd
import numpy as np


def get_df_for_scores(session_dict, measure_attribute='score'):
    data_keys, values = [], []
    for key, unit_array in session_dict.items():
        for value in unit_array:
            data_keys.append(key)
            values.append(value)
    df = pd.DataFrame({'dataset':data_keys, measure_attribute:values})
    return df


# Helpers for Readout-Position Color Plot

# smooth color interpolation
def lerp(x, a, b):
    return a + x * (b-a)

# smooth color interpolation
def serp(x, a, b):
    return a + (3*x**2 - 2*x**3) * (b-a)

def get_color(x, y, a, b, c, d, interpolation="linear"):
    if interpolation == "linear":
        img = np.array([lerp(y, lerp(x, a[i], b[i]),
                                 lerp(x, c[i], d[i])) for i in range(3)])
    else:
        img = np.array([serp(y, serp(x, a[i], b[i]),
                                 serp(x, c[i], d[i])) for i in range(3)])
    return img

def get_base_colormap(c1, c2, c3, c4, n=200, interpolation="linear"):

    w = h = n
    verts = [c1,c2,c3,c4]
    img = np.empty((h,w,3), np.uint8)
    for y in range(h):
        for x in range(w):
            img[y,x] = get_color(x/w, y/h, *verts, interpolation=interpolation)
    return img


class ColorMap2D:
    def __init__(self, cmap_array, transpose=False, reverse_x=False, reverse_y=False, xclip=None, yclip=None):
        """
        Maps two 2D array to an RGB color space based on a given reference image.
        Args:
            filename (str): reference image to read the x-y colors from
            rotate (bool): if True, transpose the reference image (swap x and y axes)
            reverse_x (bool): if True, reverse the x scale on the reference
            reverse_y (bool): if True, reverse the y scale on the reference
            xclip (tuple): clip the image to this portion on the x scale; (0,1) is the whole image
            yclip  (tuple): clip the image to this portion on the y scale; (0,1) is the whole image
        """
        self._img = cmap_array
        if transpose:
            self._img = self._img.transpose()
        if reverse_x:
            self._img = self._img[::-1,:,:]
        if reverse_y:
            self._img = self._img[:,::-1,:]
        if xclip is not None:
            imin, imax = map(lambda x: int(self._img.shape[0] * x), xclip)
            self._img = self._img[imin:imax,:,:]
        if yclip is not None:
            imin, imax = map(lambda x: int(self._img.shape[1] * x), yclip)
            self._img = self._img[:,imin:imax,:]
        if issubclass(self._img.dtype.type, np.integer):
            self._img = self._img / 255.0

        self._width = len(self._img)
        self._height = len(self._img[0])

        self._range_x = (0, 1)
        self._range_y = (0, 1)


    @staticmethod
    def _scale_to_range(u: np.ndarray, u_min: float, u_max: float) -> np.ndarray:
        return (u - u_min) / (u_max - u_min)

    def _map_to_x(self, val: np.ndarray) -> np.ndarray:
        xmin, xmax = self._range_x
        val = self._scale_to_range(val, xmin, xmax)
        rescaled = (val * (self._width - 1))
        return rescaled.astype(int)

    def _map_to_y(self, val: np.ndarray) -> np.ndarray:
        ymin, ymax = self._range_y
        val = self._scale_to_range(val, ymin, ymax)
        rescaled = (val * (self._height - 1))
        return rescaled.astype(int)

    def __call__(self, val_x, val_y):
        """
        Take val_x and val_y, and associate the RGB values
        from the reference picture to each item. val_x and val_y
        must have the same shape.
        """
        if val_x.shape != val_y.shape:
            raise ValueError(f'x and y array must have the same shape, but have {val_x.shape} and {val_y.shape}.')
        self._range_x = (np.amin(val_x), np.amax(val_x))
        self._range_y = (np.amin(val_y), np.amax(val_y))
        x_indices = self._map_to_x(val_x)
        y_indices = self._map_to_y(val_y)
        i_xy = np.stack((x_indices, y_indices), axis=-1)
        rgb = np.zeros((*val_x.shape, 3))
        for indices in np.ndindex(val_x.shape):
            img_indices = tuple(i_xy[indices])
            rgb[indices] = self._img[img_indices]
        return rgb

    def generate_cbar(self, nx=100, ny=100):
        "generate an image that can be used as a 2D colorbar"
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        return self.__call__(*np.meshgrid(x, y))