"""Classical noise attenuation pipeline, duplicated from the Aalto ITSP book chapter
"Noise attenuation" (https://speechprocessingbook.aalto.fi/Enhancement/Noise_attenuation.html).

Reproduces the non-neural sections of the notebook:
  1. Energy-threshold VAD gating
  2. Gating with hysteresis
  3. Gating with fade-in/fade-out slopes
  4. Bin-wise gating
  5. Noise model estimation (VAD-mean and minimum statistics)
  6. Spectral subtraction
  7. Wiener / MMSE gain
  8. Noise filling (soft threshold)

Plots are saved to plots/, enhanced audio to results/.
"""

import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from helper_functions import stft, istft, halfsinewindow

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS = os.path.join(ROOT, 'plots')
RESULTS = os.path.join(ROOT, 'results')
os.makedirs(PLOTS, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)


def save_wav(name, x, fs):
    """Save a float signal as 16-bit PCM WAV in results/."""
    path = os.path.join(RESULTS, name)
    wavfile.write(path, fs, x.astype(np.int16))
    print(f'wrote {path}')


def save_plot(name):
    """Save the current matplotlib figure to plots/ and close it."""
    path = os.path.join(PLOTS, name)
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close('all')
    print(f'wrote {path}')


# --- Parameters (as in the notebook) ---
window_length_ms = 30
window_step_ms = 15

filename = os.path.join(ROOT, 'data', 'sounds', 'enhancement_test.wav')

# --- Read noisy audio ---
fs, data = wavfile.read(filename)
data = data[:]

window_length = int(window_length_ms * fs / 2000) * 2
window_step_samples = int(window_step_ms * fs / 1000)
windowing_function = halfsinewindow(window_length)

spectrogram_matrix = stft(data, fs,
                          window_length_ms=window_length_ms,
                          window_step_ms=window_step_ms,
                          windowing_function=windowing_function)
window_count = spectrogram_matrix.shape[0]
fft_length = spectrogram_matrix.shape[1]
length_in_s = window_count * window_step_samples / fs
t = np.arange(0, window_count, 1) * window_step_samples / fs

plt.figure(figsize=[12, 6])
plt.subplot(211)
tt = np.arange(0, len(data), 1) / fs
plt.plot(tt, data)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Waveform of noisy audio')
plt.axis([0, len(data) / fs, 1.05 * np.min(data), 1.05 * np.max(data)])
plt.subplot(212)
plt.imshow(20 * np.log10(np.abs(spectrogram_matrix.T)),
           origin='lower', aspect='auto',
           extent=[0, length_in_s, 0, fs / 2000])
plt.axis([0, length_in_s, 0, 8])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (kHz)')
plt.title('Spectrogram of noisy audio')
save_plot('01_noisy_input.png')

# --- 1. VAD through energy thresholding ---
frame_energy = np.sum(np.abs(spectrogram_matrix) ** 2, axis=1)
frame_energy_dB = 10 * np.log10(frame_energy)
mean_energy_dB = np.mean(frame_energy_dB)

threshold_dB = mean_energy_dB + 3.
speech_active = frame_energy_dB > threshold_dB

spectrogram_thresholded = spectrogram_matrix * np.expand_dims(speech_active, axis=1)
data_thresholded = istft(spectrogram_thresholded, fs,
                         window_length_ms=window_length_ms,
                         window_step_ms=window_step_ms,
                         windowing_function=windowing_function)
save_wav('02_thresholded.wav', data_thresholded, fs)

# --- 2. Hysteresis ---
hysteresis_time_ms = 300
hysteresis_time = int(hysteresis_time_ms / window_step_ms)

speech_active_hysteresis = np.zeros([window_count])
for window_ix in range(window_count):
    range_start = max(0, window_ix - hysteresis_time)
    speech_active_hysteresis[window_ix] = np.max(speech_active[range(range_start, window_ix + 1)])

spectrogram_hysteresis = spectrogram_matrix * np.expand_dims(speech_active_hysteresis, axis=1)
data_hysteresis = istft(spectrogram_hysteresis, fs,
                        window_length_ms=window_length_ms,
                        window_step_ms=window_step_ms,
                        windowing_function=windowing_function)
save_wav('03_hysteresis.wav', data_hysteresis, fs)

# --- 3. Fade-in and fade-out ---
fade_in_time_ms = 50
fade_out_time_ms = 300
fade_in_time = int(fade_in_time_ms / window_step_ms)
fade_out_time = int(fade_out_time_ms / window_step_ms)

speech_active_sloped = np.zeros([window_count])
for frame_ix in range(window_count):
    if speech_active_hysteresis[frame_ix]:
        range_start = max(0, frame_ix - fade_in_time)
        speech_active_sloped[frame_ix] = np.mean(speech_active_hysteresis[range(range_start, frame_ix + 1)])
    else:
        range_start = max(0, frame_ix - fade_out_time)
        speech_active_sloped[frame_ix] = np.mean(speech_active_hysteresis[range(range_start, frame_ix + 1)])

spectrogram_sloped = spectrogram_matrix * np.expand_dims(speech_active_sloped, axis=1)
data_sloped = istft(spectrogram_sloped, fs,
                    window_length_ms=window_length_ms,
                    window_step_ms=window_step_ms,
                    windowing_function=windowing_function)
save_wav('04_sloped.wav', data_sloped, fs)

plt.figure(figsize=[12, 4])
normalized_frame_energy = frame_energy_dB - mean_energy_dB
plt.plot(t, normalized_frame_energy, label='Frame energy')
plt.plot(t, speech_active * 6, label='Threshold')
plt.plot(t, speech_active_hysteresis * 5, label='Hysteresis')
plt.plot(t, speech_active_sloped * 4, label='Sloped')
plt.legend()
plt.xlabel('Time (s)')
plt.title('VAD variants')
save_plot('02_vad_variants.png')

# --- 4. Bin-wise gating ---
hysteresis_time_ms = 100
hysteresis_time = int(hysteresis_time_ms / window_step_ms)
fade_in_time = int(30 / window_step_ms)
fade_out_time = int(60 / window_step_ms)

bin_energy_dB = 20 * np.log10(np.abs(spectrogram_matrix) + 1e-12)
bin_mean_dB = np.mean(bin_energy_dB, axis=0)
bin_active = bin_energy_dB > np.expand_dims(bin_mean_dB + 3., axis=0)

spectrogram_binwise = np.zeros(spectrogram_matrix.shape, dtype=complex)
for bin_ix in range(fft_length):
    active_hyst = np.zeros(window_count)
    for window_ix in range(window_count):
        range_start = max(0, window_ix - hysteresis_time)
        active_hyst[window_ix] = np.max(bin_active[range(range_start, window_ix + 1), bin_ix])
    gain = np.zeros(window_count)
    for frame_ix in range(window_count):
        if active_hyst[frame_ix]:
            range_start = max(0, frame_ix - fade_in_time)
        else:
            range_start = max(0, frame_ix - fade_out_time)
        gain[frame_ix] = np.mean(active_hyst[range(range_start, frame_ix + 1)])
    spectrogram_binwise[:, bin_ix] = spectrogram_matrix[:, bin_ix] * gain

data_binwise = istft(spectrogram_binwise, fs,
                     window_length_ms=window_length_ms,
                     window_step_ms=window_step_ms,
                     windowing_function=windowing_function)
save_wav('05_binwise.wav', data_binwise, fs)

# --- 5. Noise model estimation ---
noise_threshold_dB = mean_energy_dB - 3.
noise_active = frame_energy_dB < noise_threshold_dB

noise_frames = spectrogram_matrix[noise_active, :]
noise_estimate = np.mean(np.abs(noise_frames) ** 2, axis=0)
noise_estimate_minimum = np.min(np.abs(spectrogram_matrix) ** 2, axis=0)

f = np.linspace(0, fs / 1000, fft_length)
plt.figure(figsize=[8, 4])
plt.plot(f, 10 * np.log10(noise_estimate), label='Mean+VAD')
plt.plot(f, 10 * np.log10(noise_estimate_minimum), label='Minimum')
plt.legend()
plt.xlabel('Frequency (kHz)')
plt.ylabel('Magnitude (dB)')
plt.title('Noise models')
save_plot('03_noise_models.png')

# --- 6. Spectral subtraction ---
energy_enhanced = np.subtract(np.abs(spectrogram_matrix) ** 2, np.expand_dims(noise_estimate, axis=0))
energy_enhanced *= (energy_enhanced > 0)  # threshold at zero
enhancement_gain = np.sqrt(energy_enhanced / (np.abs(spectrogram_matrix) ** 2))
spectrogram_enhanced = spectrogram_matrix * enhancement_gain

data_enhanced = istft(spectrogram_enhanced, fs,
                      window_length_ms=window_length_ms,
                      window_step_ms=window_step_ms,
                      windowing_function=windowing_function)
save_wav('06_spectral_subtraction.wav', data_enhanced, fs)

# --- 7. Wiener / MMSE gain ---
mmse_gain = energy_enhanced / (np.abs(spectrogram_matrix) ** 2)
spectrogram_enhanced_mmse = spectrogram_matrix * mmse_gain

data_enhanced_mmse = istft(spectrogram_enhanced_mmse, fs,
                           window_length_ms=window_length_ms,
                           window_step_ms=window_step_ms,
                           windowing_function=windowing_function)
save_wav('07_wiener_mmse.wav', data_enhanced_mmse, fs)

# --- 8. Noise filling ---
noisefill_threshold_dB = -60  # dBs below average noise
noisefill_level = (10 ** (noisefill_threshold_dB / 10)) * noise_estimate

energy_enhanced = np.abs(spectrogram_matrix) ** 2 - noise_estimate
energy_noisefill = noisefill_level + 0.5 * (energy_enhanced - noisefill_level
                                            + np.abs(energy_enhanced - noisefill_level))
noisefill_gain = np.sqrt(energy_noisefill / (np.abs(spectrogram_matrix) ** 2))
spectrogram_enhanced_noisefill = spectrogram_matrix * noisefill_gain

data_enhanced_noisefill = istft(spectrogram_enhanced_noisefill, fs,
                                window_length_ms=window_length_ms,
                                window_step_ms=window_step_ms,
                                windowing_function=windowing_function)
save_wav('08_noisefill.wav', data_enhanced_noisefill, fs)

plt.figure(figsize=[12, 6])
plt.subplot(211)
plt.imshow(20 * np.log10(np.abs(spectrogram_matrix.T)),
           origin='lower', aspect='auto',
           extent=[0, length_in_s, 0, fs / 2000])
plt.axis([0, length_in_s, 0, 8])
plt.title('Noisy')
plt.ylabel('Frequency (kHz)')
plt.subplot(212)
plt.imshow(20 * np.log10(np.abs(spectrogram_enhanced.T) + 1e-6),
           origin='lower', aspect='auto',
           extent=[0, length_in_s, 0, fs / 2000])
plt.axis([0, length_in_s, 0, 8])
plt.title('Enhanced (spectral subtraction)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (kHz)')
save_plot('04_enhanced_spectrograms.png')

print('done')
