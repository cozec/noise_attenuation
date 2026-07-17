"""Test the pretrained MossFormer2_SE_48K model on our noisy test audio.

Uses ClearerVoice-Studio (https://github.com/modelscope/ClearerVoice-Studio)
with the pretrained MossFormer2_SE_48K speech enhancement model to enhance
data/sounds/enhancement_test.wav, saving the result alongside the other
methods for comparison.

The model operates at 48 kHz; ClearVoice resamples the 44.1 kHz input on the
way in, and the enhanced output is resampled back to the input rate before
saving so all results share the same sample rate.

Requires the Python 3.11 venv: .venv-dfn/bin/python src/mossformer2_test.py
(the checkpoint downloads from HuggingFace on first run)
Outputs:
    results/mossformer2_enhanced.wav
    plots/mossformer2_spectrograms.png
"""

import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from clearvoice import ClearVoice
from scipy.io import wavfile
from scipy.signal import resample_poly

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SR = 48000


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

    fs, noisy = wavfile.read(filename)
    print(f'loaded {filename}: {len(noisy)} samples at {fs} Hz')

    cv = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])
    enhanced_48k = cv(input_path=filename, online_write=False)
    enhanced_48k = np.asarray(enhanced_48k).squeeze()
    print(f'model output: {enhanced_48k.shape} at {MODEL_SR} Hz, '
          f'max |x| = {np.max(np.abs(enhanced_48k)):.3f}')

    # Resample back to the input rate; model output is float in [-1, 1]
    out = resample_poly(enhanced_48k.astype(float), fs, MODEL_SR)
    out = out[:len(noisy)] * 32767
    out_wav = os.path.join(results_dir, 'mossformer2_enhanced.wav')
    wavfile.write(out_wav, fs, np.clip(out, -32768, 32767).astype(np.int16))
    print(f'wrote {out_wav}')

    # Spectrogram comparison plot
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
    plt.imshow(spectrogram_db(out, fs).T,
               origin='lower', aspect='auto',
               extent=[0, length_in_s, 0, fs / 2000])
    plt.axis([0, length_in_s, 0, 8])
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (kHz)')
    plt.title('MossFormer2-enhanced spectrogram')
    plt.tight_layout()
    path = os.path.join(plots_dir, 'mossformer2_spectrograms.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
