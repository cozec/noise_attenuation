"""Standalone Wiener filtering for noise attenuation.

Self-contained duplicate of the Wiener/MMSE section of the Aalto ITSP book
chapter "Noise attenuation"
(https://speechprocessingbook.aalto.fi/Enhancement/Noise_attenuation.html).

Pipeline: load noisy audio -> STFT -> VAD-based noise estimation ->
Wiener gain -> ISTFT -> save enhanced audio -> plots.

Usage:
    python wiener_filtering.py [input.wav]

Defaults to data/sounds/enhancement_test.wav. Outputs:
    results/wiener_enhanced.wav
    plots/wiener_waveforms.png
    plots/wiener_spectrograms.png
    plots/wiener_noise_model.png
"""

import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft
from scipy.io import wavfile

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


# --- Wiener filtering ---

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


def wiener_filter(spectrogram, noise_estimate):
    """Apply the Wiener gain to a noisy spectrogram.

    The Wiener gain is the ratio of estimated speech power to noisy power,
    G = max(|Y|^2 - N, 0) / |Y|^2, applied per time-frequency bin.
    """
    noisy_power = np.abs(spectrogram) ** 2
    speech_power = noisy_power - np.expand_dims(noise_estimate, axis=0)
    speech_power *= (speech_power > 0)  # threshold at zero
    gain = speech_power / noisy_power
    return spectrogram * gain, gain


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

    # 4. Wiener gain + 5. reconstruction
    spectrogram_enhanced, gain = wiener_filter(spectrogram, noise_estimate)
    data_enhanced = istft(spectrogram_enhanced, fs, window_length_ms, window_step_ms,
                          fft_size=fft_size)

    out_wav = os.path.join(results_dir, 'wiener_enhanced.wav')
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
    plt.title('Wiener-enhanced waveform')
    plt.axis([0, t[-1], 1.05 * np.min(data), 1.05 * np.max(data)])
    plt.tight_layout()
    path = os.path.join(plots_dir, 'wiener_waveforms.png')
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
    plt.title('Wiener-enhanced spectrogram')
    plt.tight_layout()
    path = os.path.join(plots_dir, 'wiener_spectrograms.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'wrote {path}')

    f = np.linspace(0, fs / 2000, fft_length)
    plt.figure(figsize=[8, 4])
    plt.plot(f, 10 * np.log10(noise_estimate + 1e-6))
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Estimated noise power spectrum')
    path = os.path.join(plots_dir, 'wiener_noise_model.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
