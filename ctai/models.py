from typing import Tuple

import torch
import torch.nn as nn
from vroc.blocks import GaussianSmoothing3d, SpatialTransformer
from vroc.registration import VrocRegistration

from ctai.blocks import DenseNet


class InpaintingNet(nn.Module):
    def __init__(
        self,
        patch_shape: Tuple[int, ...],
        growth_rate: int = 16,
        dense_net_class=DenseNet,
        n_blocks: int = 2,
        n_pconv_blocks: int = 2,
        n_block_layers: int = 4,
        registration_iterations: Tuple[int, ...] = (128,),
        registration_scale_factors: Tuple[float, ...] = (1.0,),
        freeze_registration_block: bool = False,
        disable_registration: bool = False,
        disable_registration_correction: bool = False,
        regularize_correction: bool = True,
        disable_conditional_image: bool = False,
        disable_inpainting_net: bool = False,
        insert_warped_conditional: bool = False,
        fuse_output: bool = True,
    ):
        super().__init__()

        self.patch_shape = patch_shape
        self.growth_rate = growth_rate
        self.n_blocks = n_blocks
        self.n_pconv_blocks = n_pconv_blocks
        self.n_block_layers = n_block_layers

        self.registration_iterations = registration_iterations
        self.registration_scale_factors = registration_scale_factors

        self.freeze_registration_block = freeze_registration_block
        self.disable_registration = disable_registration
        self.disable_registration_correction = disable_registration_correction
        self.regularize_correction = regularize_correction
        self.disable_conditional_image = disable_conditional_image
        self.disable_inpainting_net = disable_inpainting_net

        self.fuse_output = fuse_output
        self.insert_warped_conditional = insert_warped_conditional

        if self.disable_conditional_image:
            self.in_channels = 1
        else:
            self.in_channels = 2
        self.out_channels = 1

        if not self.disable_inpainting_net:
            self.dense_net = dense_net_class(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                growth_rate=self.growth_rate,
                n_blocks=self.n_blocks,
                n_block_layers=self.n_block_layers,
                n_pconv_blocks=self.n_pconv_blocks,
            )
        else:
            self.dense_net = None

        if not self.disable_registration_correction:
            self.vector_field_updater = nn.Sequential(
                nn.Conv3d(
                    # 3 DVF channels, 2x1 image channels
                    in_channels=5,
                    out_channels=32,
                    kernel_size=(3, 3, 3),
                    padding="same",
                    bias=False,
                ),
                nn.InstanceNorm3d(num_features=32),
                nn.Mish(),
                nn.Conv3d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=(3, 3, 3),
                    padding="same",
                    bias=False,
                ),
                nn.InstanceNorm3d(num_features=64),
                nn.Mish(),
                nn.Conv3d(
                    in_channels=64,
                    out_channels=32,
                    kernel_size=(3, 3, 3),
                    padding="same",
                    bias=False,
                ),
                nn.InstanceNorm3d(num_features=32),
                nn.Mish(),
                nn.Conv3d(
                    in_channels=32,
                    out_channels=3,
                    kernel_size=(3, 3, 3),
                    padding="same",
                    bias=True,
                ),
            )

        else:
            self.vector_field_updater = None

        if not self.disable_registration:
            self.vroc = VrocRegistration()

            if self.freeze_registration_block:
                for param in self.registration_block.parameters():
                    param.requires_grad = False
        else:
            self.registration_block = None

    def load_reg_block_state(self, filepath: str, device: str = "cuda"):
        assert self.registration_block is not None

        loaded_state = torch.load(filepath, map_location=device)
        del loaded_state["model_state_dict"]["_regularization_layer.weight_x"]
        del loaded_state["model_state_dict"]["_regularization_layer.weight_y"]
        del loaded_state["model_state_dict"]["_regularization_layer.weight_z"]
        self.registration_block.load_state_dict(
            loaded_state["model_state_dict"], strict=False
        )

    @property
    def config(self):
        return dict(
            growth_rate=self.growth_rate,
            n_blocks=self.n_blocks,
            n_pconv_blocks=self.n_pconv_blocks,
            n_block_layers=self.n_block_layers,
            registration_iterations=self.registration_iterations,
            freeze_registration_block=self.freeze_registration_block,
            disable_registration=self.disable_registration,
            disable_registration_correction=self.disable_registration_correction,
            disable_conditional_image=self.disable_conditional_image,
            disable_inpainting_net=self.disable_inpainting_net,
            fuse_output=self.fuse_output,
            insert_warped_conditional=self.insert_warped_conditional,
        )

    def forward(self, image, mask, conditional):
        valid_image = image * mask
        artifact_mask = 1 - mask

        if not self.disable_registration:
            params = {
                "iterations": 200,
                "tau": 2.25,
                "tau_level_decay": 0.0,
                "tau_iteration_decay": 0.0,
                "sigma_x": 1.25,
                "sigma_y": 1.25,
                "sigma_z": 1.25,
                "sigma_level_decay": 0.0,
                "sigma_iteration_decay": 0.0,
                "n_levels": 3,
            }

            if not self.training:
                params = {
                    "iterations": 800,
                    "tau": 2.0,
                    "tau_level_decay": 0.0,
                    "tau_iteration_decay": 0.0,
                    "sigma_x": 1.25,
                    "sigma_y": 1.25,
                    "sigma_z": 1.25,
                    "sigma_level_decay": 0.0,
                    "sigma_iteration_decay": 0.0,
                    "n_levels": 3,
                }
                print(f"{params=}")

            registration_result = self.vroc.register(
                moving_image=conditional,
                fixed_image=image,
                moving_mask=None,
                fixed_mask=mask,
                register_affine=True,
                affine_enable_rotation=True,
                affine_enable_scaling=True,
                affine_enable_shearing=True,
                affine_enable_translation=True,
                force_type="demons",
                gradient_type="passive",
                valid_value_range=(-1024, 3071),
                early_stopping_delta=0.00001,
                early_stopping_window=None,
                debug=False,
                default_parameters=params,
                return_as_tensor=True,
            )

            warped_conditional = registration_result.warped_moving_image.clone()
            vector_field = registration_result.composed_vector_field.clone()

            if not self.disable_registration_correction:
                # update/correct DVF at artifact region
                masked_image = image * mask
                stacked = torch.cat(
                    (vector_field, masked_image, warped_conditional), dim=1
                )

                vector_field_correction = self.vector_field_updater(stacked)

                corrected_vector_field = vector_field + vector_field_correction

                regularizer = GaussianSmoothing3d(
                    sigma=(params["sigma_x"], params["sigma_y"], params["sigma_z"])
                ).to(corrected_vector_field)
                corrected_vector_field = regularizer(corrected_vector_field)

                corrected_warped_conditional = SpatialTransformer()(
                    image=conditional, transformation=corrected_vector_field
                )
            else:
                corrected_warped_conditional = warped_conditional
                corrected_vector_field = vector_field

            torch.cuda.empty_cache()

        else:
            corrected_warped_conditional = conditional
            corrected_vector_field = None

        if self.insert_warped_conditional:
            conditional_inserted = (
                valid_image + artifact_mask * corrected_warped_conditional
            )
            images = torch.cat(
                (corrected_warped_conditional, conditional_inserted), dim=1
            )
        else:
            images = torch.cat((corrected_warped_conditional, valid_image), dim=1)

        if self.disable_conditional_image:
            corrected_warped_conditional = None
            images = images[:, 1:]  # only valid image, no conditional

        # residual output:
        if not self.disable_inpainting_net:
            # with torch.autocast(device_type="cuda", enabled=True):
            inpainted, mask_out = self.dense_net(images, mask)
            inpainted = 2 * torch.sigmoid(inpainted) - 1
        else:
            inpainted = torch.zeros_like(valid_image)
            mask_out = None

        if self.fuse_output:
            if self.disable_conditional_image:
                final_image = valid_image + artifact_mask * inpainted
            else:
                final_image = valid_image + artifact_mask * (
                    corrected_warped_conditional + inpainted
                )
        else:
            if self.disable_conditional_image:
                final_image = inpainted
            else:
                final_image = corrected_warped_conditional + inpainted

        return (
            final_image,
            mask_out,
            corrected_warped_conditional,
            corrected_vector_field,
        )
