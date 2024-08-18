from external.CNO2d import CNO2d


def build_cno(nChannels, device, s, N_layers, N_res, N_res_neck, channel_multiplier):
    """
    Build and return a CNO model.

    Parameters:
    nChannels (int): Number of input and output channels.
    device (torch.device): Device to run the model on.
    s (int): spatial dimension
    N_layers (int): Number of (D) or (U) blocks in the network
    N_res (int): Number of (R) blocks per level (except the neck)
    N_res_neck (int): Number of (R) blocks in the neck
    channel_multiplier (int): How the number of channels evolve

    Returns:
    model: The CNO model.
    """
    model = CNO2d(
        in_dim=nChannels,
        out_dim=nChannels,
        size=s,
        N_layers=N_layers,
        N_res=N_res,
        N_res_neck=N_res_neck,
        channel_multiplier=channel_multiplier,
        use_bn=False,
    ).to(device)
    return model
