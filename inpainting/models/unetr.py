from monai.networks.nets import UNETR

def build_unetr(nChannels, device, target_img_shape, hidden_size=768, feature_size=16, num_heads=12):
    """
    Build and return a UNETR model.

    Parameters:
    nChannels (int): Number of input and output channels.
    device (torch.device): Device to run the model on.
    target_img_shape (tuple): Target image shape.
    hidden_size (int): Size of the hidden layers.
    feature_size (int): Feature size.
    num_heads (int): Number of attention heads.

    Returns:
    model: The UNETR model.
    """
    model = UNETR(
        in_channels=nChannels,
        out_channels=nChannels,
        spatial_dims=2,
        img_size=target_img_shape,
        feature_size=feature_size,
        hidden_size=hidden_size,
        mlp_dim=hidden_size * 4,
        num_heads=num_heads,
        norm_name="instance",
        res_block=True,
        dropout_rate=0,
    ).to(device)
    return model