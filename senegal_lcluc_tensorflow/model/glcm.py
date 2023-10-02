import math
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.util import apply_parallel, img_as_ubyte
from skimage.transform import pyramid_expand
from typing import List, Tuple


class GLCM(object):

    pi4 = math.pi / 4

    def __init__(self, **kwargs) -> None:
        self.grid_size = kwargs.get("grid_size", 3)
        self.overlap = kwargs.get("overlap", 2)
        self.bin_size = kwargs.get("bin_size", 4)
        self.distances = kwargs.get("distances", [1, 2, 3, 4])
        self.angles = kwargs.get(
            "angles", [0, self.pi4, 2 * self.pi4, 3 * self.pi4])
        self.features = kwargs.get("features", ["homogeneity"])

    def _rescale(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - X.min()) / (X.max() - X.min())

        return img_as_ubyte(Xs) // self.bin_size

    def _unpack(self,
                image: np.ndarray,
                padding: List[Tuple[int, int]],
                offset: int):
        fdata = image[0:: self.grid_size, offset:: self.grid_size].reshape(
            [s // self.grid_size for s in image.shape]
        )
        upscaled = pyramid_expand(
            fdata,
            upscale=self.grid_size,
            sigma=None,
            order=1,
            mode="reflect",
            cval=0,
            channel_axis=None,
            preserve_range=False,
        )
        return np.pad(upscaled, padding, mode="edge")

    def _glcm_feature(self, patch: np.ndarray):
        levels = 256 // self.bin_size
        if patch.size == 1:
            return np.zeros_like(patch, dtype=np.float)
        glcm = graycomatrix(
            patch,
            self.distances,
            self.angles,
            levels,
            symmetric=True,
            normed=True
        )
        rv: np.ndarray = np.full(patch.shape, 0, dtype=np.float)
        for iF, feature in enumerate(self.features):
            i0, i1 = (
                self.overlap + iF // self.bin_size,
                self.overlap + iF % self.bin_size,
            )
            if feature == "mean":
                rv[i0, i1] = self._graycoprops(glcm, feature)[0, 0]
            else:
                rv[i0, i1] = graycoprops(glcm, feature)[0, 0]
        return rv

    def _apply_test(self, P: np.ndarray, prop: str = "mean"):
        pass

    def _graycoprops(self, P: np.ndarray, prop="mean"):
        """
            Parameters
            ----------
            P : ndarray
                Input array. `P` is the gray-level co-occurrence histogram
                for which to compute the specified property. The value
                `P[i,j,d,theta]` is the number of times that gray-level j
                occurs at a distance d and at an angle theta from
                gray-level i.
            prop : {'contrast', 'dissimilarity', 'homogeneity', 'energy', \
                    'correlation', 'ASM'}, optional
                The property of the GLCM to compute. The default is 'contrast'.
            Returns
            -------
            results : 2-D ndarray
                2-dimensional array. `results[d, a]` is the property 'prop' for
                the d'th distance and the a'th angle.
        """
        (num_level, num_level2, num_dist, num_angle) = P.shape
        if num_level != num_level2:
            raise ValueError("num_level and num_level2 must be equal.")
        if num_dist <= 0:
            raise ValueError("num_dist must be positive.")
        if num_angle <= 0:
            raise ValueError("num_angle must be positive.")

        # normalize each GLCM
        P = P.astype(np.float64)
        glcm_sums = np.sum(P, axis=(0, 1), keepdims=True)
        glcm_sums[glcm_sums == 0] = 1
        P /= glcm_sums

        if prop == "mean":
            I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
            results = np.sum(I * P, axis=(0, 1))
        else:
            raise ValueError(f"{prop} is an invalid property")

        return results

    def compute_band_features(
        self, image: np.ndarray
    ) -> List[np.ndarray]:  # input_data: dims = [ y, x ]
        block_size = [(s // self.grid_size) *
                      self.grid_size for s in image.shape]
        padding = [(0, image.shape[i] - block_size[i]) for i in (0, 1)]
        raw_image: np.ndarray = image[: block_size[0], : block_size[1]]
        image: np.ndarray = self._rescale(raw_image)
        features = apply_parallel(
            self._glcm_feature,
            image,
            chunks=self.grid_size,
            depth=self.overlap,
            mode="reflect",
        )
        return [
            self._unpack(features, padding, iF)
            for iF, _ in enumerate(self.features)
        ]