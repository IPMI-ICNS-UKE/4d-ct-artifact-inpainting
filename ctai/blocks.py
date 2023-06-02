import torch
import torch.nn as nn

from ctai.layers import PartialConv2d, PartialConv3d


class DualSeparableConv3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        kx, ky, kz = kernel_size

        self.conv_1 = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=(kx, 1, 1),
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            dilation=dilation,
            bias=False,
            groups=in_channels,
        )
        self.conv_2 = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=(1, ky, 1),
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            dilation=dilation,
            bias=False,
            groups=in_channels,
        )
        self.conv_3 = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=(1, 1, kz),
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            dilation=dilation,
            bias=False,
            groups=in_channels,
        )
        self.pointwise = nn.Conv3d(
            in_channels, out_channels, kernel_size=(1, 1, 1), bias=bias
        )

        self._chained_convs = nn.Sequential(
            self.conv_1, self.conv_2, self.conv_3, self.pointwise
        )

    def init_weights(self):
        nn.init.dirac_(self.conv_1.weight)
        nn.init.dirac_(self.conv_2.weight)
        nn.init.dirac_(self.conv_3.weight)

        nn.init.dirac_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, x):
        return self._chained_convs(x)


class PConvBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_chanels = in_channels
        self.out_channels = out_channels

        self._partial_conv = PartialConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3, 3),
            padding="same",
            multi_channel=True,
            return_mask=True,
        )
        self._image_layers = nn.Sequential(nn.BatchNorm3d(out_channels), nn.ReLU())

    def forward(self, image, mask, **kwargs):
        image_out, mask_out = self._partial_conv(image, mask)
        image_out = self._image_layers(image_out)

        return image_out, mask_out


class CondPConvBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_chanels = in_channels
        self.out_channels = out_channels

        self._conditional_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding="same",
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding="same",
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )

        self._partial_conv = PartialConv2d(
            in_channels=in_channels + out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding="same",
            multi_channel=True,
            return_mask=True,
        )
        self._image_layers = nn.Sequential(nn.BatchNorm2d(out_channels), nn.ReLU())

    def forward(self, image, mask, conditional):
        conditional_out = self._conditional_layers(conditional)

        stacked_images = torch.cat((conditional_out, image), dim=1)

        flat_mask = mask.any(dim=1, keepdims=True).to(torch.float32)

        mask = torch.repeat_interleave(
            flat_mask, self.in_chanels + self.out_channels, dim=1
        )

        image_out, mask_out = self._partial_conv(stacked_images, mask)
        image_out = self._image_layers(image_out)

        return image_out, mask_out, conditional_out


class PConvNormActivation(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3, 3),
        padding="same",
    ):
        super().__init__()
        conv = PartialConv3d
        self._partial_conv = conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            multi_channel=True,
            return_mask=True,
            bias=False,
        )
        self._norm = nn.InstanceNorm3d(out_channels)
        self._activation = nn.Mish()

    def forward(self, image, mask):
        image_out, mask_out = self._partial_conv(image, mask)
        image_out = self._norm(image_out)
        image_out = self._activation(image_out)

        return image_out, mask_out


class ConvNormActivation(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=(3, 3, 3), padding="same"
    ):
        super().__init__()
        conv = nn.Conv3d
        self._conv = conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self._norm = nn.InstanceNorm3d(out_channels)
        self._activation = nn.Mish()

    def forward(self, image, mask):
        image_out = self._conv(image)
        image_out = self._norm(image_out)
        image_out = self._activation(image_out)

        return image_out, mask


class CondPConvBlock3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_convs: int = 1,
        convolve_conditional: bool = True,
    ):
        super().__init__()

        self.in_chanels = in_channels
        self.out_channels = out_channels
        self.n_convs = n_convs
        self.convolve_conditional = convolve_conditional

        if convolve_conditional:
            self._conditional_layers = nn.Sequential(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(3, 3, 3),
                    padding="same",
                ),
                nn.InstanceNorm3d(num_features=out_channels),
                nn.ReLU(),
                nn.Conv3d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=(3, 3, 3),
                    padding="same",
                ),
                nn.InstanceNorm3d(num_features=out_channels),
                nn.ReLU(),
            )
            pconv_in_channels = in_channels + out_channels
        else:
            self._conditional_layers = None
            pconv_in_channels = in_channels + 1

        for i_conv in range(self.n_convs):
            self.add_module(
                f"partial_conv_block{i_conv + 1}",
                PConvNormActivation(
                    in_channels=pconv_in_channels,
                    out_channels=out_channels,
                    kernel_size=(3, 3, 3),
                    padding="same",
                ),
            )
            pconv_in_channels = out_channels

    def forward(self, image, mask, conditional):
        if self.convolve_conditional:
            conditional_out = self._conditional_layers(conditional)
        else:
            conditional_out = conditional

        stacked_images = torch.cat((conditional_out, image), dim=1)

        flat_mask = mask.any(dim=1, keepdims=True).to(torch.float32)

        n_stacked_channels = stacked_images.shape[1]
        mask = torch.repeat_interleave(flat_mask, n_stacked_channels, dim=1)

        image_out, mask_out = stacked_images, mask
        for i_conv in range(self.n_convs):
            partial_conv_block = self.get_submodule(f"partial_conv_block{i_conv + 1}")
            image_out, mask_out = partial_conv_block(image_out, mask_out)

        return image_out, mask_out, conditional_out, {}


class ResidualDenseBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        growth_rate: int = 32,
        n_layers: int = 4,
        partial_conv: bool = True,
    ):
        super().__init__()

        self.growth_rate = growth_rate
        self.n_layers = n_layers
        self.partial_conv = partial_conv
        init_in_channels = in_channels

        for i_layer in range(self.n_layers):
            if self.partial_conv:
                conv = PConvNormActivation(
                    in_channels=in_channels,
                    out_channels=self.growth_rate,
                    kernel_size=(3, 3, 3),
                    padding="same",
                )
                name = f"partial_conv_block{i_layer + 1}"
            else:
                conv = ConvNormActivation(
                    in_channels=in_channels,
                    out_channels=self.growth_rate,
                    kernel_size=(3, 3, 3),
                    padding="same",
                )
                name = f"conv_block{i_layer + 1}"

            self.add_module(name, conv)

            in_channels = (i_layer + 1) * self.growth_rate + init_in_channels

        self.feature_fuse = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=self.growth_rate,
                kernel_size=(1, 1, 1),
                bias=False,
            ),
            nn.InstanceNorm3d(num_features=self.growth_rate),
            nn.Mish(),
        )

    def expand_mask(self, mask, image):
        n_stacked_channels = image.shape[1]
        flat_mask = mask.any(dim=1, keepdims=True).to(torch.float32)
        mask = torch.repeat_interleave(flat_mask, n_stacked_channels, dim=1)

        return mask

    def forward(self, image, mask):
        outputs = []
        image_in = image
        mask_out = mask
        image_out = image
        for i_layer in range(self.n_layers):
            if self.partial_conv:
                name = f"partial_conv_block{i_layer + 1}"
            else:
                name = f"conv_block{i_layer + 1}"

            layer = self.get_submodule(name)
            stacked = torch.cat((image_in, *outputs), dim=1)

            # match mask
            mask_out = self.expand_mask(mask_out, stacked)

            image_out, mask_out = layer(stacked, mask_out)
            outputs.append(image_out)

        stacked = torch.cat((image_in, *outputs), dim=1)
        mask_out = self.expand_mask(mask_out, stacked)

        image_out = self.feature_fuse(stacked)

        image_out = image_in + image_out

        return image_out, mask_out


class ChainedDenseNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        growth_rate: int = 32,
        n_blocks: int = 2,
        n_block_layers: int = 4,
        n_pconv_blocks: int = 4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.growth_rate = growth_rate
        self.n_blocks = n_blocks
        self.n_block_layers = n_block_layers
        self.n_pconv_blocks = n_pconv_blocks

        self._conditional_dense_net = DenseNet(
            in_channels=self.in_channels,  # warped condtional & valid artifact image
            out_channels=self.growth_rate,
            growth_rate=self.growth_rate,
            n_blocks=self.n_blocks,
            n_block_layers=self.n_block_layers,
            n_pconv_blocks=0,
        )
        self._norm = nn.InstanceNorm3d(self.growth_rate)
        self._activation = nn.Mish()

        self._inpainting_dense_net = DenseNet(
            in_channels=1 + self.growth_rate,  # valid artifact image & above features
            out_channels=self.out_channels,
            growth_rate=self.growth_rate,
            n_blocks=self.n_blocks,
            n_block_layers=self.n_block_layers,
            n_pconv_blocks=self.n_pconv_blocks,
        )

    def forward(self, image, mask):
        # image = (warped condtional, valid artifact image)
        if image.shape[1] == 2:
            artifact_image = image[:, 1:2]
        elif image.shape[1] == 1:
            # no conditional given
            artifact_image = image
        else:
            raise ValueError

        conditional_features, _ = self._conditional_dense_net(image, mask)
        conditional_features = self._norm(conditional_features)
        conditional_features = self._activation(conditional_features)

        image_out, mask_out = self._inpainting_dense_net(
            torch.cat((artifact_image, conditional_features), dim=1), mask
        )

        return image_out, mask_out


class DenseNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        growth_rate: int = 32,
        n_blocks: int = 2,
        n_block_layers: int = 4,
        n_pconv_blocks: int = 4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.n_blocks = n_blocks
        self.n_block_layers = n_block_layers
        self.n_pconv_blocks = n_pconv_blocks

        self.feature_extend = nn.Sequential(
            nn.Conv3d(
                in_channels=self.in_channels,
                out_channels=self.growth_rate,
                kernel_size=(1, 1, 1),
                bias=False,
            ),
            nn.InstanceNorm3d(num_features=self.growth_rate),
            nn.Mish(),
        )

        for i_block in range(self.n_blocks):
            is_partial_conv = i_block < self.n_pconv_blocks
            self.add_module(
                f"residual_dense_block{i_block + 1}",
                ResidualDenseBlock(
                    in_channels=self.growth_rate,
                    growth_rate=self.growth_rate,
                    n_layers=self.n_block_layers,
                    partial_conv=is_partial_conv,
                ),
            )

        self.feature_fuse = nn.Conv3d(
            in_channels=self.in_channels + self.n_blocks * self.growth_rate,
            out_channels=out_channels,
            kernel_size=(1, 1, 1),
            bias=True,
        )

        # init with (warped) conditional as output
        nn.init.dirac_(self.feature_fuse.weight)
        nn.init.zeros_(self.feature_fuse.bias)

    def expand_mask(self, mask, image):
        n_stacked_channels = image.shape[1]
        flat_mask = mask.any(dim=1, keepdims=True).to(torch.float32)
        mask = torch.repeat_interleave(flat_mask, n_stacked_channels, dim=1)

        return mask

    def forward(self, image, mask):
        image_outputs = [image]
        masks_outputs = [mask]

        mask_out = self.expand_mask(mask, image)
        image_out = self.feature_extend(image)

        for i_block in range(self.n_blocks):
            layer = self.get_submodule(f"residual_dense_block{i_block + 1}")

            image_out, mask_out = layer(image_out, mask_out)
            image_outputs.append(image_out)
            masks_outputs.append(mask_out)

        mask_out = torch.cat(masks_outputs, dim=1)
        stacked = torch.cat(image_outputs, dim=1)
        mask_out = self.expand_mask(mask_out, stacked)

        image_out = self.feature_fuse(stacked)

        return image_out, mask_out
