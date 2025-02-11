import torch

def kl_divergence(spectrum1, spectrum2):
    spectrum1, spectrum2 = torch.as_tensor(spectrum1), torch.as_tensor(spectrum2)

    spectrum1 = spectrum1 / torch.sum(spectrum1)
    spectrum2 = spectrum2 / torch.sum(spectrum2)

    kl_div = torch.nn.functional.kl_div(spectrum1.log(), spectrum2, reduction='sum')
    return kl_div

def radial_average(image):
    y, x = torch.meshgrid(torch.arange(0, image.size(-2)),
                          torch.arange(0, image.size(-1)), indexing='ij')
    center = torch.tensor([image.size(-2) / 2, image.size(-1) / 2])

    r = torch.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.type(torch.long)

    radial_sum = torch.zeros(r.max() + 1)
    count = torch.zeros(r.max() + 1)

    for i in range(image.size(-2)):
        for j in range(image.size(-1)):
            radial_sum[r[i, j]] += image[i, j]
            count[r[i, j]] += 1

    radialprofile = radial_sum / count
    nyq = int(image.size(-2) / 2)
    return radialprofile[1:nyq+1]  # Skip the zero distance bin and larger than Nyquist limit

def spectral_div_channelwise(true, pred):
    kl_div_sum = 0.0

    for channel in range(true.size(0)):
        true_channel = true[channel].type(torch.complex64)
        pred_channel = pred[channel].type(torch.complex64)

        fft_true = torch.fft.fftshift(torch.fft.fft2(true_channel))
        fft_pred = torch.fft.fftshift(torch.fft.fft2(pred_channel))

        energy_true = fft_true * torch.conj(fft_true)
        energy_pred = fft_pred * torch.conj(fft_pred)

        energy_spectrum_true = 0.5 * radial_average(energy_true.real)
        energy_spectrum_pred = 0.5 * radial_average(energy_pred.real)

        kl_div_sum += kl_divergence(energy_spectrum_true, energy_spectrum_pred)

    kl = kl_div_sum / true.size(0)
    return kl, energy_spectrum_true, energy_spectrum_pred