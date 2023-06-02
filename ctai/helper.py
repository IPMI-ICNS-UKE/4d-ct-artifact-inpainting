from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk

logger = logging.getLogger(__name__)


def resize(image: np.ndarray, output_shape: Tuple[int, ...], order: int = 1):
    factors = np.asarray(image.shape, dtype=np.float32) / np.asarray(
        output_shape, dtype=np.float32
    )

    coord_arrays = [
        factors[i] * (np.arange(d) + 0.5) - 0.5 for i, d in enumerate(output_shape)
    ]

    coord_map = np.stack(np.meshgrid(*coord_arrays, sparse=False, indexing="ij"))
    image = image.astype(np.float32)
    out = ndi.map_coordinates(image, coord_map, order=order, mode="nearest")

    return out


def resize_isotropic(
    image: np.ndarray, image_spacing: Tuple[float, ...], order: int = 1
):
    output_shape = tuple(
        int(round(sp * sh)) for (sp, sh) in zip(image_spacing, image.shape)
    )
    return resize(image=image, output_shape=output_shape, order=order)


def resample_image_spacing(
    image: sitk.Image,
    new_spacing: Tuple[float, float, float],
    resampler=sitk.sitkLinear,
    default_voxel_value=0.0,
):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / new_spacing[2]))),
    ]
    resampled_img = sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        resampler,
        image.GetOrigin(),
        new_spacing,
        image.GetDirection(),
        default_voxel_value,
        image.GetPixelID(),
    )
    return resampled_img


def bounding_box_3d(image: np.ndarray, padding: int = 0) -> Tuple[slice, slice, slice]:
    x = np.any(image, axis=(1, 2))
    y = np.any(image, axis=(0, 2))
    z = np.any(image, axis=(0, 1))

    if isinstance(padding, int):
        padding = (padding,) * 3

    x_min, x_max = np.where(x)[0][[0, -1]]
    y_min, y_max = np.where(y)[0][[0, -1]]
    z_min, z_max = np.where(z)[0][[0, -1]]

    x_min, x_max = max(x_min - padding[0], 0), min(x_max + padding[0], image.shape[0])
    y_min, y_max = max(y_min - padding[1], 0), min(y_max + padding[1], image.shape[1])
    z_min, z_max = max(z_min - padding[2], 0), min(z_max + padding[2], image.shape[2])

    return np.index_exp[x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1]


def robust_bounding_box_3d(
    image: np.ndarray, bbox_range: Tuple[float, float] = (0.01, 0.99), padding: int = 0
) -> Tuple[slice, slice, slice]:
    x = np.cumsum(image.sum(axis=(1, 2)))
    y = np.cumsum(image.sum(axis=(0, 2)))
    z = np.cumsum(image.sum(axis=(0, 1)))

    x = x / x[-1]
    y = y / y[-1]
    z = z / z[-1]

    x_min, x_max = np.searchsorted(x, bbox_range[0]), np.searchsorted(x, bbox_range[1])
    y_min, y_max = np.searchsorted(y, bbox_range[0]), np.searchsorted(y, bbox_range[1])
    z_min, z_max = np.searchsorted(z, bbox_range[0]), np.searchsorted(z, bbox_range[1])

    x_min, x_max = max(x_min - padding, 0), min(x_max + padding, image.shape[0])
    y_min, y_max = max(y_min - padding, 0), min(y_max + padding, image.shape[1])
    z_min, z_max = max(z_min - padding, 0), min(z_max + padding, image.shape[2])

    return np.index_exp[x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1]


def get_roi_from_masks(masks: List[Optional[np.ndarray]], padding: int = 0):
    merged_mask = None
    for i, mask in enumerate(masks):
        if mask is None:
            continue
        if i == 0:
            merged_mask = mask
        else:
            merged_mask = np.logical_or(mask, merged_mask)

    if merged_mask is not None:
        roi_bbox = robust_bounding_box_3d(
            merged_mask, bbox_range=(0.01, 0.99), padding=padding
        )
    else:
        roi_bbox = None

    return roi_bbox, merged_mask


def nearest_factor_pow_2(
    value: int,
    factors: Tuple[int, ...] = (2, 3, 5, 6, 7, 9),
    min_exponent: int | None = None,
    max_value: int | None = None,
    allow_smaller_value: bool = False,
) -> int:
    factors = np.array(factors)
    upper_exponents = np.ceil(np.log2(value / factors))
    lower_exponents = upper_exponents - 1

    if min_exponent:
        upper_exponents[upper_exponents < min_exponent] = np.inf
        lower_exponents[lower_exponents < min_exponent] = np.inf

    def get_distances(
        factors: Tuple[int, ...], exponents: Tuple[int, ...], max_value: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pow2_values = factors * 2**exponents
        if max_value:
            mask = pow2_values <= max_value
            pow2_values = pow2_values[mask]
            factors = factors[mask]
            exponents = exponents[mask]

        return np.abs(pow2_values - value), factors, exponents

    distances, _factors, _exponents = get_distances(
        factors=factors, exponents=upper_exponents, max_value=max_value
    )
    if len(distances) == 0:
        if allow_smaller_value:
            distances, _factors, _exponents = get_distances(
                factors=factors, exponents=lower_exponents, max_value=max_value
            )
        else:
            raise RuntimeError("Could not find a value")

    if len(distances):
        nearest_factor = _factors[np.argmin(distances)]
        nearest_exponent = _exponents[np.argmin(distances)]
    else:
        # nothing found
        pass

    return int(nearest_factor * 2**nearest_exponent)


def pad_bounding_box_to_pow_2(
    bounding_box: Tuple[slice, ...],
    factors: Tuple[int, ...] = (2, 3, 5, 6, 7, 9),
    reference_shape: Tuple[int, ...] | None = None,
) -> tuple[slice, ...]:
    if any([b.step and b.step > 1 for b in bounding_box]):
        raise NotImplementedError("Only step size of 1 for now")

    n_dim = len(bounding_box)
    bbox_shape = tuple(b.stop - b.start for b in bounding_box)
    if reference_shape:
        print(bounding_box)
        print(bbox_shape, reference_shape)
        padding = tuple(
            nearest_factor_pow_2(
                s, factors=factors, max_value=r, allow_smaller_value=True
            )
            - s
            for s, r in zip(bbox_shape, reference_shape)
        )
    else:
        padding = tuple(
            nearest_factor_pow_2(s, factors=factors) - s for s in bbox_shape
        )

    padded_bbox = []
    for i_axis in range(n_dim):
        padding_left = padding[i_axis] // 2
        padding_right = padding[i_axis] - padding_left

        padded_slice = slice(
            bounding_box[i_axis].start - padding_left,
            bounding_box[i_axis].stop + padding_right,
        )

        if padded_slice.start < 0:
            padded_slice = slice(
                0,
                padded_slice.stop - padded_slice.start,
            )

        padded_bbox.append(padded_slice)
    return tuple(padded_bbox)


def rescale_range(
    values: np.ndarray, input_range: Tuple, output_range: Tuple, clip: bool = True
):
    in_min, in_max = input_range
    out_min, out_max = output_range
    rescaled = (((values - in_min) * (out_max - out_min)) / (in_max - in_min)) + out_min
    if clip:
        return np.clip(rescaled, out_min, out_max)
    return rescaled


def crop_or_pad(image: np.ndarray, target_shape: Tuple[int, ...], pad_value=0.0):
    valid_content_slicing = [
        slice(None, None),
    ] * image.ndim

    for i_axis in range(image.ndim):
        if target_shape[i_axis] is not None:
            if image.shape[i_axis] < target_shape[i_axis]:
                # pad
                padding = target_shape[i_axis] - image.shape[i_axis]
                padding_left = padding // 2
                padding_right = padding - padding_left

                pad_width = [(0, 0)] * image.ndim
                pad_width[i_axis] = (padding_left, padding_right)
                image = np.pad(
                    image, pad_width, mode="constant", constant_values=pad_value
                )
                valid_content_slicing[i_axis] = slice(padding_left, -padding_right)
            elif image.shape[i_axis] > target_shape[i_axis]:
                cropping = image.shape[i_axis] - target_shape[i_axis]
                cropping_left = cropping // 2
                cropping_right = cropping - cropping_left

                cropping_slicing = [
                    slice(None, None),
                ] * image.ndim
                cropping_slicing[i_axis] = slice(cropping_left, -cropping_right)
                image = image[tuple(cropping_slicing)]

    return image, tuple(valid_content_slicing)
