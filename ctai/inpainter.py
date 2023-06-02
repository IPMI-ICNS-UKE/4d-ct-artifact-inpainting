import logging
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn as nn
from tqdm import tqdm

from ctai.helper import (
    bounding_box_3d,
    crop_or_pad,
    nearest_factor_pow_2,
    pad_bounding_box_to_pow_2,
    rescale_range,
)
from ctai.stitcher import PatchStitcher

logger = logging.getLogger(__name__)


class Inpainter(ABC):
    def _pad(self, image: np.array, pad_value: float = 0.0):
        padded_shape = []
        for s in image.shape:
            padded_shape.append(nearest_factor_pow_2(s))

        image, valid_slicing = crop_or_pad(
            image=image, target_shape=tuple(padded_shape), pad_value=pad_value
        )

        return image, valid_slicing

    def _add_axes(self, image: np.array):
        return image[np.newaxis, np.newaxis]

    def scale_hounsfield(self, image: np.ndarray):
        return rescale_range(image, input_range=(-1024, 3071), output_range=(0, 1))

    def rescale_hounsfield(self, image: np.ndarray):
        return rescale_range(image, input_range=(0, 1), output_range=(-1024, 3071))

    @abstractmethod
    def inpaint(
        self,
        artifact_image: np.ndarray,
        conditional_image: np.ndarray,
        artifact_mask: np.ndarray,
    ):
        pass


class NoopInpainter3D(Inpainter):
    def inpaint(
        self,
        *,
        artifact_image: np.ndarray,
        conditional_image: np.ndarray,
        artifact_mask: np.ndarray,
        **kwargs,
    ):
        misc = {}
        return artifact_image, misc


class InsertConditionalInpainter3D(Inpainter):
    def inpaint(
        self,
        *,
        artifact_image: np.ndarray,
        conditional_image: np.ndarray,
        artifact_mask: np.ndarray,
        **kwargs,
    ):
        misc = {}
        inpainted = (
            1 - artifact_mask
        ) * artifact_image + artifact_mask * conditional_image
        return inpainted, misc


class Inpainter3D(Inpainter):
    def __init__(
        self, inpaining_model: nn.Module, device="cuda", autocast: bool = False
    ):
        self.inpainting_model = inpaining_model.to(device=device)
        self.inpainting_model.eval()

        self.device = device
        self.autocast = autocast

    def inpaint(
        self,
        *,
        artifact_image: np.ndarray,
        conditional_image: np.ndarray,
        artifact_mask: np.ndarray,
        patch_shape: Tuple[int, int, int] = (64, 128, 128),
        stride_factor: float = 2.0,
        restrict_to_artifact: bool = True,
        roi_mask: Optional[np.ndarray] = None,
        roi_mask_padding: int = 0,
        fuse_images: bool = True,
        return_misc: bool = False,
        verbose: bool = False,
        slicings: Sequence[Tuple[slice, slice, slice]] = None,
        use_roi_bbox_as_slicing: bool = False,
        do_center_crop: bool = True,
        boundary_smoothing_sigma: float = 0.0,
        **kwargs,
    ) -> Tuple[np.ndarray, dict]:
        assert artifact_image.shape == conditional_image.shape == artifact_mask.shape
        # stride = tuple(int(ps // stride_factor) for ps in patch_shape)
        margin = tuple(ps // 8 for ps in patch_shape)

        artifact_image = rescale_range(
            artifact_image, input_range=(-1024, 3071), output_range=(0, 1)
        )
        conditional_image = rescale_range(
            conditional_image, input_range=(-1024, 3071), output_range=(0, 1)
        )

        artifact_mask = artifact_mask > 0

        extraction_mask = artifact_mask.copy()
        if roi_mask is not None:
            roi_bbox = bounding_box_3d(roi_mask, padding=max(margin) + roi_mask_padding)
            print(f"{roi_bbox=}")
            bbox_mask = np.zeros_like(extraction_mask)
            bbox_mask[roi_bbox] = 1
            extraction_mask = extraction_mask * bbox_mask

            if use_roi_bbox_as_slicing:
                extraction_bbox = bounding_box_3d(extraction_mask, padding=(0, 0, 32))
                extraction_bbox = pad_bounding_box_to_pow_2(extraction_bbox)

                # bbox or max of shape
                extraction_bbox = list(extraction_bbox)
                for i, s in enumerate(extraction_bbox):
                    if s.stop > extraction_mask.shape[i]:
                        extraction_bbox[i] = slice(s.start, extraction_mask.shape[i])

                extraction_bbox = tuple(extraction_bbox)
                slicing = extraction_bbox
                print(f"{extraction_bbox=}")

        stitcher = PatchStitcher(artifact_image.shape, color_axis=None)
        conditional_stitcher = PatchStitcher(artifact_image.shape, color_axis=None)
        dvf_stitcher = PatchStitcher(artifact_image.shape, color_axis=None)

        artifact_image = self._add_axes(artifact_image)
        conditional_image = self._add_axes(conditional_image)
        artifact_mask = self._add_axes(artifact_mask)
        inpainting_mask = np.logical_not(artifact_mask)

        _artifact_image = torch.as_tensor(artifact_image, dtype=torch.float32).to(
            device=self.device
        )
        _conditional_image = torch.as_tensor(conditional_image, dtype=torch.float32).to(
            device=self.device
        )
        _inpainting_mask = torch.as_tensor(inpainting_mask, dtype=torch.float32).to(
            device=self.device
        )
        if verbose:
            slicings = tqdm(slicings)

        with torch.no_grad():
            for spatial_slicing in slicings:
                print(spatial_slicing)
                slicing = (slice(None), slice(None)) + spatial_slicing

                artifact_image_patch = _artifact_image[slicing]
                conditional_image_patch = _conditional_image[slicing]
                inpainting_mask_patch = _inpainting_mask[slicing]

                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=self.autocast
                ):
                    (
                        inpainted_image_patch,
                        output_mask_patch,
                        warped_conditional_patch,
                        dvf_patch,
                    ) = self.inpainting_model(
                        artifact_image_patch,
                        inpainting_mask_patch,
                        conditional_image_patch,
                    )

                inpainted_image_patch = inpainted_image_patch.detach().cpu().numpy()
                inpainted_image_patch = inpainted_image_patch.squeeze(axis=(0, 1))

                if return_misc:
                    if warped_conditional_patch is not None:
                        warped_conditional_patch = (
                            warped_conditional_patch.detach().cpu().numpy()
                        )
                        warped_conditional_patch = warped_conditional_patch.squeeze(
                            axis=(0, 1)
                        )
                    else:
                        warped_conditional_patch = np.zeros_like(inpainted_image_patch)

                    if dvf_patch is not None:
                        dvf_patch = dvf_patch.detach().cpu().numpy()
                        dvf_patch = dvf_patch.squeeze(axis=0)[2]
                    else:
                        dvf_patch = np.zeros_like(warped_conditional_patch)

                if do_center_crop:
                    center_crop = tuple(
                        slice(m, ps - m) for ps, m in zip(patch_shape, margin)
                    )
                    stitcher_slicing = tuple(
                        slice(s.start + m, s.stop - m)
                        for s, m in zip(spatial_slicing, margin)
                    )
                else:
                    center_crop = Ellipsis
                    stitcher_slicing = spatial_slicing

                stitcher.add_patch(
                    data=np.asarray(
                        inpainted_image_patch[center_crop], dtype=np.float32
                    ),
                    slicing=stitcher_slicing,
                )
                if return_misc:
                    conditional_stitcher.add_patch(
                        data=np.asarray(
                            warped_conditional_patch[center_crop], dtype=np.float32
                        ),
                        slicing=stitcher_slicing,
                    )
                    dvf_stitcher.add_patch(
                        data=np.asarray(dvf_patch[center_crop], dtype=np.float32),
                        slicing=stitcher_slicing,
                    )

        inpainted = stitcher.calculate_mean(default_value=0.0)
        inpainted = self.rescale_hounsfield(inpainted)

        if fuse_images:
            artifact_image = self.rescale_hounsfield(artifact_image)
            artifact_mask = artifact_mask.squeeze()
            artifact_mask = (artifact_mask > 0.0).astype(np.float32)
            if boundary_smoothing_sigma > 0.0:
                artifact_mask = ndi.gaussian_filter(
                    artifact_mask.astype(np.float32),
                    sigma=(0, 0, boundary_smoothing_sigma),
                )
            inpainted = artifact_mask * inpainted + (1 - artifact_mask) * artifact_image

        if return_misc:
            conditional_warped = conditional_stitcher.calculate_mean(default_value=0.0)
            conditional_warped = self.rescale_hounsfield(conditional_warped)
            dvf = dvf_stitcher.calculate_mean(default_value=0.0)
            misc = {
                "conditional_warped": conditional_warped.squeeze(),
                "dvf": dvf.squeeze(),
            }
        else:
            misc = {}

        return inpainted.squeeze(), misc
