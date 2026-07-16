"""Test the pretrained DeepFilterNet model on our noisy test audio.

Uses the pretrained DeepFilterNet3 model (https://github.com/Rikorose/DeepFilterNet)
to enhance data/sounds/enhancement_test.wav and saves the result alongside the
classical methods for comparison.

DeepFilterNet operates at 48 kHz; load_audio resamples the 44.1 kHz input on
the way in, and the enhanced output is resampled back to the input rate before
saving so all results share the same sample rate.

Requires the Python 3.11 venv: .venv-dfn/bin/python src/deepfilternet_test.py
Outputs:
    results/deepfilternet_enhanced.wav
    plots/deepfilternet_spectrograms.png
"""

import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from df.enhance import enhance, init_df, load_audio
from df.io import resample
from scipy.io import wavfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def spectrogram_db(x, fs, window_length_ms=30, window_step_ms=15):
    """Magnitude spectrogram in dB for plotting (frames x bins)."""
    window_length = int(window_length_ms * fs / 2000) * 2
    window_step = int(window_step_ms * fs / 1000)
    window = np.sin(np.pi * np.arange(0.5, window_length, 1) / window_length)
    window_count = int((len(x) - window_length) / window_step) + 1
    frames = np.stack([x[k * window_step:k * window_step + window_length] * window
                       for k in range(window_count)])
    return 20 * np.log10(np.abs(np.fft.rfft(frames, axis=1)) + 1e-6)


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, 'data', 'sounds', 'enhancement_test.wav')
    results_dir = os.path.join(ROOT, 'results')
    plots_dir = os.path.join(ROOT, 'plots')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Load pretrained model (DeepFilterNet3 by default, downloaded on first run)
    model, df_state, _ = init_df()
    model_sr = df_state.sr()

    # Load and resample input to the model's 48 kHz
    audio, meta = load_audio(filename, sr=model_sr)
    print(f'loaded {filename}: {meta.num_frames} samples at {meta.sample_rate} Hz '
          f'-> resampled to {model_sr} Hz')

    # Enhance
    enhanced = enhance(model, df_state, audio)

    # Resample back to the input rate and save
    enhanced_out = resample(enhanced, model_sr, meta.sample_rate)
    # load_audio returns floats in [-1, 1]; scale back to int16 range
    out = enhanced_out.squeeze(0).numpy() * 32767
    out_wav = os.path.join(results_dir, 'deepfilternet_enhanced.wav')
    wavfile.write(out_wav, meta.sample_rate, np.clip(out, -32768, 32767).astype(np.int16))
    print(f'wrote {out_wav}')

    # Spectrogram comparison plot
    fs, noisy = wavfile.read(filename)
    length_in_s = len(noisy) / fs
    plt.figure(figsize=[12, 6])
    plt.subplot(211)
    plt.imshow(spectrogram_db(noisy.astype(float), fs).T,
               origin='lower', aspect='auto',
               extent=[0, length_in_s, 0, fs / 2000])
    plt.axis([0, length_in_s, 0, 8])
    plt.ylabel('Frequency (kHz)')
    plt.title('Noisy spectrogram')
    plt.subplot(212)
    plt.imshow(spectrogram_db(out.astype(float), fs).T,
               origin='lower', aspect='auto',
               extent=[0, length_in_s, 0, fs / 2000])
    plt.axis([0, length_in_s, 0, 8])
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (kHz)')
    plt.title('DeepFilterNet-enhanced spectrogram')
    plt.tight_layout()
    path = os.path.join(plots_dir, 'deepfilternet_spectrograms.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
