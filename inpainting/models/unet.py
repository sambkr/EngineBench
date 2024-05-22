from monai.networks.nets import UNet

def build_unet(nChannels, device, channels=(64, 128, 256, 512, 1024), strides=(2, 2, 2, 2)):
    """
    Build and return a UNet model.

    Parameters:
    nChannels (int): Number of input and output channels.
    device (torch.device): Device to run the model on.
    channels (tuple): Tuple of channel sizes for each layer.
    strides (tuple): Tuple of stride sizes for each layer.

    Returns:
    model: The UNet model.
    """
    model = UNet(
        spatial_dims=2,
        in_channels=nChannels,
        out_channels=nChannels,
        channels=channels,
        strides=strides,
        kernel_size=3,
        up_kernel_size=3,
        act="ReLU",
        norm="instance",
        dropout=0.0,
        bias=True,
        adn_ordering="NDA",
    ).to(device)
    return model