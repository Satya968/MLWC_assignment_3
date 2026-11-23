"""Turn complex channel grids into the single-channel images the CNNs expect.

Each complex grid is split into its real and imaginary parts, which are then
treated as two independent grayscale images and stacked along the batch axis:

    (N, N_SUB, N_SYM) complex   ->   (2N, 1, N_SUB, N_SYM) float32
                                      [ all real parts ; all imag parts ]

The same convolutional network therefore sees the real-part image and then
the imaginary-part image, matching the original ChannelNet recipe.

Images are scaled so the network operates on unit-variance data.  The scale is
the standard deviation of the *target* real/imag images; because the real and
imaginary parts each carry half of E|H|^2 = 1, dividing by this scale makes the
mean-squared error on the normalised images equal to the complex-channel MSE
up to the constant 2*scale^2, which we undo explicitly when reporting.
"""
from __future__ import annotations

import numpy as np

from . import settings as S
from .pilots import least_squares
from .classical import ls_interpolate

# train/val get their own disjoint noise streams; test uses the shared default
_NOISE_OFFSET = {"train": 100_000, "val": 200_000}


def pilot_input(H: np.ndarray, pilots: int, snr_db: float, split: str) -> np.ndarray:
    """LS + interpolation image for one split, with the right noise stream."""
    if split == "test":
        seed = None                                   # shared test realisation
    else:
        seed = S.NOISE_SEED + _NOISE_OFFSET[split] + int(round(10 * snr_db))
    return ls_interpolate(least_squares(H, pilots, snr_db, seed=seed), pilots)


def to_images(H: np.ndarray) -> np.ndarray:
    """(N, N_SUB, N_SYM) complex -> (2N, 1, N_SUB, N_SYM) float32."""
    stacked = np.concatenate([H.real, H.imag], axis=0)
    return stacked[:, np.newaxis, :, :].astype(np.float32)


def from_images(img: np.ndarray) -> np.ndarray:
    """Inverse of :func:`to_images`: (2N,1,H,W) -> (N,H,W) complex64."""
    half = len(img) // 2
    real = img[:half, 0]
    imag = img[half:, 0]
    return (real + 1j * imag).astype(np.complex64)


class ImageScaler:
    """Fit the normalisation scale on a target array and apply it reversibly."""

    def __init__(self, scale: float):
        self.scale = float(scale)

    @classmethod
    def fit(cls, H_target: np.ndarray) -> "ImageScaler":
        return cls(to_images(H_target).std())

    def encode(self, H: np.ndarray) -> np.ndarray:
        """Complex grid -> normalised image tensor (numpy)."""
        return to_images(H) / self.scale

    def decode(self, img: np.ndarray) -> np.ndarray:
        """Normalised image tensor -> complex grid."""
        return from_images(img * self.scale)
