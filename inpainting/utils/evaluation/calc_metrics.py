import numpy as np
import torch

from utils.evaluation import RI, MI, relative_l2


def calc_metrics(
    snap, recon, centre_mask, edge_mask, nChannels, spectral_energy, n, horiz
):

    # Vector metrics
    metrics_c = np.zeros((3,))
    metrics_e = np.zeros((3,))

    metrics_c[0] = RI(snap[centre_mask == 0], recon[centre_mask == 0])
    metrics_c[1] = MI(snap[centre_mask == 0], recon[centre_mask == 0])
    metrics_c[2] = relative_l2(snap[centre_mask == 0], recon[centre_mask == 0])

    metrics_e[0] = RI(snap[edge_mask != 0], recon[edge_mask != 0])
    metrics_e[1] = MI(snap[edge_mask != 0], recon[edge_mask != 0])
    metrics_e[2] = relative_l2(snap[edge_mask != 0], recon[edge_mask != 0])

    # Spectral metrics
    energy_matrix = torch.zeros_like(spectral_energy[0])

    centre_snap = torch.clone(snap)
    centre_snap[centre_mask != 0] = 0  # Get centre region
    centre_pred = torch.clone(recon)
    centre_pred[centre_mask != 0] = 0

    for j in range(nChannels):
        fft_tr = torch.fft.fftshift(torch.fft.fft2(centre_snap[j]))  # True
        fft_pr = torch.fft.fftshift(torch.fft.fft2(centre_pred[j]))  # Pred

        energy_matrix[0, j, ...] = fft_tr * torch.conj(fft_tr)
        energy_matrix[1, j, ...] = fft_pr * torch.conj(fft_pr)

    snap = snap - centre_snap
    recon = recon - centre_pred
    if horiz:
        snap[..., n:-n, :] = 0
        recon[..., n:-n, :] = 0
    else:
        snap[..., n:-n] = 0
        recon[..., n:-n] = 0

    for j in range(nChannels):
        fft_tr = torch.fft.fftshift(torch.fft.fft2(snap[j]))  # True
        fft_pr = torch.fft.fftshift(torch.fft.fft2(recon[j]))  # Pred

        energy_matrix[2, j, ...] = fft_tr * torch.conj(fft_tr)
        energy_matrix[3, j, ...] = fft_pr * torch.conj(fft_pr)

    return metrics_c, metrics_e, energy_matrix
