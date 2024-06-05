from dataclasses import dataclass
from typing import Union, Optional

import numpy as np
from elias.config import Config


def to_spherical(cartesian_points: np.ndarray) -> np.ndarray:
    x = cartesian_points[..., 0]
    y = cartesian_points[..., 1]
    z = cartesian_points[..., 2]

    radius = np.linalg.norm(cartesian_points, axis=-1, ord=2)
    theta = np.arctan2(np.sqrt(x * x + y * y), z)
    phi = np.arctan2(y, x)

    return np.stack([radius, theta, phi], axis=-1)


def to_cartesian(spherical_coordinates: np.ndarray) -> np.ndarray:
    radius = spherical_coordinates[..., 0]
    theta = spherical_coordinates[..., 1]
    phi = spherical_coordinates[..., 2]

    sin_theta = np.sin(theta)
    x = radius * np.cos(phi) * sin_theta
    y = radius * np.sin(phi) * sin_theta
    z = radius * np.cos(theta)

    return np.stack([x, y, z], axis=-1)


@dataclass
class QuantizerConfig(Config):
    min_values: Union[np.ndarray, float]
    max_values: Union[np.ndarray, float]
    bits: int
    mask_value: Optional[np.ndarray] = 0
    separate_mask: bool = True


class Quantizer:

    def __init__(self,
                 min_values: Union[np.ndarray, float],
                 max_values: Union[np.ndarray, float],
                 bits: int,
                 mask_value: Optional[np.ndarray] = 0,
                 separate_mask: bool = True):
        if mask_value is None:
            # If no mask should be used, separate_mask is not doing anything
            separate_mask = False

        self._min_values = min_values
        self._max_values = max_values
        self._bits = bits
        self._mask_value = mask_value
        self._separate_mask = separate_mask

        self._mask_offset = 1 if separate_mask else 0  # Reserve bin 0 for mask
        self._n_buckets = 2 ** self._bits
        self._scale_factor = (self._n_buckets - 1 - self._mask_offset) / (self._max_values - self._min_values)

        self._min_values = np.array(self._min_values, dtype=np.float32)
        self._max_values = np.array(self._max_values, dtype=np.float32)
        self._scale_factor = np.array(self._scale_factor, dtype=np.float32)

    def encode(self, values: np.ndarray) -> np.ndarray:
        mask = None
        if self._mask_value is not None:
            mask = values != self._mask_value
            if len(mask.shape) > 2:
                mask = mask.any(axis=-1)

        scaled_values = (np.maximum(0,
                                    values - self._min_values) * self._scale_factor) + self._mask_offset  # Reserve bin 0 for mask
        assert scaled_values.min() >= self._mask_offset
        assert scaled_values.max() < self._n_buckets

        if self._mask_value is not None:
            scaled_values[~mask] = 0

        quantized_values = scaled_values.round().astype(np.uint8 if self._bits == 8 else np.uint16)

        return quantized_values

    def decode(self, quantized_values: np.ndarray) -> np.ndarray:
        mask = None
        if self._mask_value is not None:
            mask = quantized_values == self._mask_value
            if len(mask.shape) > 2:
                mask = mask.all(axis=-1)

        values = quantized_values.astype(np.float32) / self._scale_factor - (
                self._mask_offset / self._scale_factor - self._min_values)

        if self._mask_value is not None:
            values[mask] = self._mask_value

        return values

    def to_config(self) -> QuantizerConfig:
        return QuantizerConfig(
            min_values=self._min_values,
            max_values=self._max_values,
            bits=self._bits,
            mask_value=self._mask_value,
            separate_mask=self._separate_mask
        )

    @classmethod
    def from_config(cls, config: QuantizerConfig) -> 'Quantizer':
        return cls(**config.to_json())


class DepthQuantizer(Quantizer):
    """
    Depth values are in [0, 2] and are quantized as 16 bit uints.
    A depth value of 0 is interpreted as "no depth" and will have its separate bin
    """

    def __init__(self, min_values: float = 0, max_values: float = 2, bits: int = 16, separate_mask: bool = True):
        super(DepthQuantizer, self).__init__(
            min_values=min_values,
            max_values=max_values,
            bits=bits,
            separate_mask=separate_mask)

    def encode(self, values: np.ndarray) -> np.ndarray:
        # Depth values > 2 are for sure outliers. Mask them
        values[values > self._max_values] = self._mask_value
        return super().encode(values)


class NormalsQuantizer(Quantizer):
    """
    Normals are 3-dim euclidean vectors with norm 1.
    They are first transformed to spherical coordinates (radius, phi, theta).
    The radius is fixed to 1, while phi is assumed to be in [1/3, pi], and theta in [-pi, pi].
    The reason that phi is not [0, pi] is that certain normals will never occur in our data as the cameras are
    in the front.
    Furthermore, a normal of [0, 0, 0] is interpreted as "no normal available" and will be encoded with a separate
    value in the quantized format.
    """

    def __init__(self,
                 min_values: np.ndarray = np.array([0, 1 / 3 * np.pi, -np.pi]),
                 max_values: np.ndarray = np.array([1, np.pi, np.pi]),
                 bits: int = 8):
        super(NormalsQuantizer, self).__init__(
            min_values=min_values,
            max_values=max_values,
            bits=bits)

    def encode(self, values: np.ndarray) -> np.ndarray:
        mask = values != 0
        if len(mask.shape) > 2:
            mask = mask.any(axis=-1)

        spherical_normal_map = to_spherical(values)
        quantized_spherical_normal_map = super().encode(spherical_normal_map)
        quantized_spherical_normal_map[mask][..., 0] = 1  # Drop radius because it is always 1

        return quantized_spherical_normal_map

    def decode(self, quantized_values: np.ndarray) -> np.ndarray:
        mask = quantized_values != 0
        if len(mask.shape) > 2:
            mask = mask.any(axis=-1)

        # normal_map = None
        spherical_normal_map = super().decode(quantized_values)
        normal_map = np.zeros_like(spherical_normal_map)
        normal_map[mask] = to_cartesian(spherical_normal_map[mask])

        return normal_map


class CanonicalCoordinatesQuantizer(Quantizer):
    """
    Canonical coordinates are in [-1, 1] and are quantized as 16 bit uint
    """

    def __init__(self,
                 min_values=np.array([-1, -1, -1.5, -0.2, -0.2]),
                 max_values=np.array([1, 1, 0.5, 0.2, 0.2]),
                 bits: int = 16,
                 mask_value: Optional[float] = None):
        super(CanonicalCoordinatesQuantizer, self).__init__(min_values, max_values, bits, mask_value=mask_value)


class FlameNormalsQuantizer(NormalsQuantizer):

    def __init__(self, bits: int = 8):
        super(FlameNormalsQuantizer, self).__init__(
            min_values=np.array([0, 0, -np.pi]),
            max_values=np.array([1, np.pi, np.pi]),
            bits=bits)


class UVQuantizer(Quantizer):

    def __init__(self, bits: int = 16):
        super(UVQuantizer, self).__init__(
            min_values=0,
            max_values=1,
            bits=bits,
        )
