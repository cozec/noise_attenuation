"""Standalone Ephraim-Malah noise attenuation (MMSE-STSA).

Implements the short-time spectral amplitude MMSE estimator of
Ephraim & Malah (1984), "Speech enhancement using a minimum mean-square
error short-time spectral amplitude estimator", IEEE Trans. ASSP 32(6),
with the decision-directed a priori SNR estimator.

Pipeline: load noisy audio -> STFT -> VAD-based noise estimation ->
Ephraim-Malah gain (decision-directed, frame by frame) -> ISTFT ->
save enhanced audio -> plots.

Per time-frequency bin (frame l, bin k), with noise power N(k):
    a posteriori SNR   gamma = |Y|^2 / N
    a priori SNR       xi    = alpha * A_prev^2 / N + (1-alpha) * max(gamma-1, 0)
    v                  v     = xi * gamma / (1 + xi)
    gain               G     = (sqrt(pi)/2) * (sqrt(v)/gamma) * exp(-v/2)
                               * [(1+v) I0(v/2) + v I1(v/2)]
where I0, I1 are modified Bessel functions and A_prev is the previous
frame's amplitude estimate G*|Y|. The exp/Bessel product is computed with
scipy's exponentially scaled i0e/i1e for numerical stability. The smoothing
alpha=0.98 is the classic choice; it is what suppresses "musical noise"
compared to spectral subtraction and Wiener filtering.

Usage:
    python ephraim_malah.py [input.wav]

Defaults to data/sounds/enhancement_test.wav. Outputs:
    results/ephraim_malah_enhanced.wav
    plots/ephraim_malah_waveforms.png
    plots/ephraim_malah_spectrograms.png
    plots/ephraim_malah_noise_model.png
"""

import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft
from scipy.io import wavfile
from scipy.special import i0e, i1e

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- STFT machinery (half-sine analysis/synthesis windows, as in the book) ---

def halfsinewindow(window_length):
    """Half-sine window for perfect-reconstruction overlap-add."""
    return np.sin(np.pi * np.arange(0.5, window_length, 1) / window_length)


def stft(data, fs, window_length_ms=30, window_step_ms=15, windowing_function=None,
         fft_size=None):
    """Short-time Fourier transform: rows are frames, columns are frequency bins.

    fft_size zero-pads each windowed frame to that length before the FFT
    (must be >= window length); defaults to the window length.
    """
    window_length = int(window_length_ms * fs / 2000) * 2
    window_step = int(window_step_ms * fs / 1000)
    if windowing_function is None:
        windowing_function = halfsinewindow(window_length)
    if fft_size is None:
        fft_size = window_length

    window_count = int((len(data) - window_length) / window_step) + 1
    spectrum_length = int(fft_size / 2) + 1
    spectrogram = np.zeros((window_count, spectrum_length), dtype=complex)

    for k in range(window_count):
        starting_position = k * window_step
        data_vector = data[starting_position:starting_position + window_length]
        spectrogram[k, :] = scipy.fft.rfft(data_vector * windowing_function, n=fft_size)

    return spectrogram


def istft(spectrogram, fs, window_length_ms=30, window_step_ms=15, windowing_function=None,
          fft_size=None):
    """Inverse STFT by windowed overlap-add.

    fft_size must match the value used in stft(); the inverse FFT frame is
    truncated back to the window length before overlap-add.
    """
    window_length = int(window_length_ms * fs / 2000) * 2
    window_step = int(window_step_ms * fs / 1000)
    if windowing_function is None:
        windowing_function = halfsinewindow(window_length)
    if fft_size is None:
        fft_size = window_length

    window_count = spectrogram.shape[0]
    total_length = (window_count - 1) * window_step + window_length
    data = np.zeros(total_length)

    for k in range(window_count):
        ix = np.arange(k * window_step, k * window_step + window_length)
        frame = scipy.fft.irfft(spectrogram[k, :], n=fft_size)[:window_length]
        data[ix] += frame * windowing_function

    return data


# --- Ephraim-Malah ---

def estimate_noise(spectrogram, threshold_offset_dB=-3.):
    """Estimate the noise power spectrum from low-energy (non-speech) frames.

    Frames whose total energy is more than |threshold_offset_dB| below the
    mean frame energy are treated as noise-only, and the noise model is the
    mean power spectrum over those frames.
    """
    frame_energy_dB = 10 * np.log10(np.sum(np.abs(spectrogram) ** 2, axis=1))
    noise_threshold_dB = np.mean(frame_energy_dB) + threshold_offset_dB
    noise_active = frame_energy_dB < noise_threshold_dB
    noise_estimate = np.mean(np.abs(spectrogram[noise_active, :]) ** 2, axis=0)
    return noise_estimate, noise_active


def ephraim_malah(spectrogram, noise_estimate, alpha=0.98, xi_min=10 ** (-25 / 10)):
    """Apply the Ephraim-Malah MMSE-STSA gain to a noisy spectrogram.

    Processes frames in order because the decision-directed a priori SNR
    depends on the previous frame's amplitude estimate. xi_min floors the
    a priori SNR (-25 dB) to bound the maximum suppression.
    """
    window_count, fft_length = spectrogram.shape
    noise_power = np.maximum(noise_estimate, 1e-12)

    gain_matrix = np.zeros((window_count, fft_length))
    amplitude_prev = np.zeros(fft_length)

    for l in range(window_count):
        noisy_power = np.abs(spectrogram[l, :]) ** 2

        gamma = noisy_power / noise_power  # a posteriori SNR
        xi = alpha * amplitude_prev ** 2 / noise_power \
            + (1 - alpha) * np.maximum(gamma - 1, 0)  # a priori SNR (decision-directed)
        xi = np.maximum(xi, xi_min)

        v = xi * gamma / (1 + xi)
        v = np.maximum(v, 1e-12)

        # exp(-v/2)*I0(v/2) = i0e(v/2), exp(-v/2)*I1(v/2) = i1e(v/2)
        gain = (np.sqrt(np.pi) / 2) * (np.sqrt(v) / np.maximum(gamma, 1e-12)) \
            * ((1 + v) * i0e(v / 2) + v * i1e(v / 2))
        gain = np.minimum(gain, 1.)

        gain_matrix[l, :] = gain
        amplitude_prev = gain * np.abs(spectrogram[l, :])

    return spectrogram * gain_matrix, gain_matrix


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, 'data', 'sounds', 'enhancement_test.wav')
    plots_dir = os.path.join(ROOT, 'plots')
    results_dir = os.path.join(ROOT, 'results')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    window_length_ms = 30
    window_step_ms = 15
    fft_size = 2048

    # 1. Load noisy audio
    fs, data = wavfile.read(filename)
    print(f'loaded {filename}: {len(data)} samples at {fs} Hz')

    # 2. STFT
    spectrogram = stft(data, fs, window_length_ms, window_step_ms, fft_size=fft_size)
    window_count, fft_length = spectrogram.shape
    window_step_samples = int(window_step_ms * fs / 1000)
    length_in_s = window_count * window_step_samples / fs

    # 3. Noise estimation
    noise_estimate, noise_active = estimate_noise(spectrogram)
    print(f'noise model from {np.sum(noise_active)}/{window_count} frames')
    print(f'spectrogram shape: {spectrogram.shape}, noise_estimate shape: {noise_estimate.shape}')

    # 4. Ephraim-Malah gain + 5. reconstruction
    spectrogram_enhanced, gain = ephraim_malah(spectrogram, noise_estimate)
    data_enhanced = istft(spectrogram_enhanced, fs, window_length_ms, window_step_ms,
                          fft_size=fft_size)

    out_wav = os.path.join(results_dir, 'ephraim_malah_enhanced.wav')
    wavfile.write(out_wav, fs, data_enhanced.astype(np.int16))
    print(f'wrote {out_wav}')

    # 6. Plots
    t = np.arange(len(data)) / fs
    t_enh = np.arange(len(data_enhanced)) / fs

    plt.figure(figsize=[12, 6])
    plt.subplot(211)
    plt.plot(t, data)
    plt.ylabel('Amplitude')
    plt.title('Noisy waveform')
    plt.axis([0, t[-1], 1.05 * np.min(data), 1.05 * np.max(data)])
    plt.subplot(212)
    plt.plot(t_enh, data_enhanced)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Ephraim-Malah-enhanced waveform')
    plt.axis([0, t[-1], 1.05 * np.min(data), 1.05 * np.max(data)])
    plt.tight_layout()
    path = os.path.join(plots_dir, 'ephraim_malah_waveforms.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'wrote {path}')

    plt.figure(figsize=[12, 6])
    plt.subplot(211)
    plt.imshow(20 * np.log10(np.abs(spectrogram.T) + 1e-6),
               origin='lower', aspect='auto',
               extent=[0, length_in_s, 0, fs / 2000])
    plt.axis([0, length_in_s, 0, 8])
    plt.ylabel('Frequency (kHz)')
    plt.title('Noisy spectrogram')
    plt.subplot(212)
    plt.imshow(20 * np.log10(np.abs(spectrogram_enhanced.T) + 1e-6),
               origin='lower', aspect='auto',
               extent=[0, length_in_s, 0, fs / 2000])
    plt.axis([0, length_in_s, 0, 8])
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (kHz)')
    plt.title('Ephraim-Malah-enhanced spectrogram')
    plt.tight_layout()
    path = os.path.join(plots_dir, 'ephraim_malah_spectrograms.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'wrote {path}')

    f = np.linspace(0, fs / 2000, fft_length)
    plt.figure(figsize=[8, 4])
    plt.plot(f, 10 * np.log10(noise_estimate + 1e-6))
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Estimated noise power spectrum')
    path = os.path.join(plots_dir, 'ephraim_malah_noise_model.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
