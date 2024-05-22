import numpy as np
import matplotlib.pyplot as plt


def ColorQuiver(x, y, u, v, auto_cbar=True, cbar_lims=(0, 0), ax=None):
    """
    Vector plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    else:
        fig = ax.figure

    c = (u**2 + v**2) ** 0.5
    scale = 15  # quiver plot arrows
    c_sparse = 1  # sparsity for imagesc
    sparse = 2  # sparsity for quiver plot
    constant_length = 30  # control arrow length
    scaledu = np.empty_like(u)
    scaledv = np.empty_like(v)
    # Perform division only where c is not zero
    scaledu[c != 0] = u[c != 0] * (constant_length / c[c != 0])
    scaledv[c != 0] = v[c != 0] * (constant_length / c[c != 0])
    scaledc = c / np.nanmax(c)
    dx = x[0, 0] - x[0, 1]  # calc grid spacing
    dy = y[0, 0] - y[1, 0]
    y_flip = np.flipud(y)  # flip y coords so that y.max() is at the top for matplotlib

    colorbar_limit = (0, c.max()) if auto_cbar else cbar_lims

    im = ax.imshow(
        c[::c_sparse, ::c_sparse],
        cmap="viridis",
        origin="upper",
        extent=[x.min(), x.max(), y.min(), y.max()],
        vmin=colorbar_limit[0],
        vmax=colorbar_limit[1],
    )

    q = ax.quiver(
        x[::sparse, ::sparse] - dx / 2,
        y_flip[::sparse, ::sparse] - dy / 2,
        scaledu[::sparse, ::sparse],
        scaledv[::sparse, ::sparse],
        angles="xy",
        scale_units="xy",
        scale=scale,
        color="white",
        edgecolor="white",
        width=0.004,
        linewidth=0.05,
        headlength=3,
        headaxislength=3,
        headwidth=3,
    )

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_xlim(x[0, 0], x[0, -1])
    ax.set_ylim(y[0, 0], y[-1, 0])
    #     ax.set_xlim(x[0, 40], x[0, -40])
    #     ax.set_ylim(y[40, 0], y[-40, 0])

    # Get the position of the current axes
    ax_pos = ax.get_position()

    # Create a new axes for the colorbar
    cbar_ax = fig.add_axes([ax_pos.x1 + 0.01, ax_pos.y0, 0.02, ax_pos.height - 0.03])
    cb = plt.colorbar(im, cax=cbar_ax)
    cb.set_label("Velocity magnitude (m/s)")


def Loader2Plot(tens, metadata):
    """
    Converts vector data from dataloader to numpy arrays. Expects CxHxW tensor
    """
    array = tens.detach().cpu().numpy()
    u = array[0, ...]
    v = array[1, ...]

    newu = u * metadata["std_u"] + metadata["mean_u"]
    newv = v * metadata["std_v"] + metadata["mean_v"]

    return newu, newv
