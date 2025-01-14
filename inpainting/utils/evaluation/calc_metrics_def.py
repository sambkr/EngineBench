import numpy as np
import torch

from utils.evaluation import RI, MI, relative_l2


def calc_metrics_def(
    snap, recon, true_mask, pred_mask, nChannels, spectral_energy
):
    # Isolate regions of image
    snap_true_locs = snap[true_mask]
    recon_true_locs = recon[true_mask]
    snap_pred_locs = snap[pred_mask]
    recon_pred_locs = recon[pred_mask]

    # Vector metrics
    metrics_c = np.zeros((3,))
    metrics_e = np.zeros((3,))

    metrics_c[0] = RI(snap_true_locs, recon_true_locs)
    metrics_c[1] = MI(snap_true_locs, recon_true_locs)
    metrics_c[2] = relative_l2(snap_true_locs, recon_true_locs)

    metrics_e[0] = RI(snap_pred_locs, recon_pred_locs)
    metrics_e[1] = MI(snap_pred_locs, recon_pred_locs)
    metrics_e[2] = relative_l2(snap_pred_locs, recon_pred_locs)

    # Spectral metrics
    energy_matrix = torch.zeros_like(spectral_energy[0])

    # Retain two dimensions
    centre_snap = torch.clone(snap)
    centre_snap[~true_mask] = 0  # Get centre region
    centre_recon = torch.clone(recon)
    centre_recon[~true_mask] = 0

    for j in range(nChannels):
        fft_tr = torch.fft.fftshift(torch.fft.fft2(centre_snap[j]))  # True
        fft_pr = torch.fft.fftshift(torch.fft.fft2(centre_recon[j]))  # Pred

        energy_matrix[0, j, ...] = fft_tr * torch.conj(fft_tr)
        energy_matrix[1, j, ...] = fft_pr * torch.conj(fft_pr)

    pred_snap = snap - centre_snap
    pred_recon = recon - centre_recon

    for j in range(nChannels):
        fft_tr = torch.fft.fftshift(torch.fft.fft2(pred_snap[j]))  # True
        fft_pr = torch.fft.fftshift(torch.fft.fft2(pred_recon[j]))  # Pred

        energy_matrix[2, j, ...] = fft_tr * torch.conj(fft_tr)
        energy_matrix[3, j, ...] = fft_pr * torch.conj(fft_pr)

    return metrics_c, metrics_e, energy_matrix
