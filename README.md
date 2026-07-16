# Noise Attenuation

## The three standalone enhancement scripts

Each script is fully self-contained (load audio → STFT → noise estimation →
gain → ISTFT → WAV + plots) and runs on `data/sounds/enhancement_test.wav` by
default, or any WAV passed as an argument:

```bash
.venv/bin/python src/spectral_subtraction.py   # -> results/spectral_subtraction_enhanced.wav
.venv/bin/python src/wiener_filtering.py       # -> results/wiener_enhanced.wav
.venv/bin/python src/ephraim_malah.py          # -> results/ephraim_malah_enhanced.wav
```

All three share the same front end: 30 ms half-sine windows with 15 ms hop,
2048-point zero-padded FFT, and a noise model `N(k)` estimated by averaging the
power spectra of low-energy frames (a simple energy VAD: frames more than 3 dB
below the mean frame energy are taken as noise-only). They differ only in the
gain applied to each time-frequency bin of the noisy spectrogram `Y`.

### 1. Spectral subtraction — `src/spectral_subtraction.py`

The oldest and simplest approach: subtract the noise power estimate from the
noisy power spectrum and rescale the magnitude accordingly.

```
G = sqrt( max(|Y|² − N, 0) / |Y|² )
```

The `max(·, 0)` threshold is needed because the noise estimate is an average —
in any single frame the actual noise can exceed it, making the difference
negative. Those randomly zeroed bins are what cause the characteristic
"musical noise" (isolated warbling tones) in the output.

![Spectral subtraction: noisy input (top) vs enhanced output (bottom)](plots/spectral_subtraction_spectrograms.png)

### 2. Wiener filter — `src/wiener_filtering.py`

The minimum mean-square error solution for a linear filter. Same subtraction,
but the gain is the ratio of estimated speech power to noisy power **without**
the square root:

```
G = max(|Y|² − N, 0) / |Y|²
```

Since `G ≤ 1`, omitting the square root makes the gain smaller — the Wiener
filter suppresses noise more aggressively than spectral subtraction at the
same SNR, at the price of slightly more speech attenuation. Both methods are
memoryless: each frame is processed independently, so musical noise remains.

![Wiener filter: noisy input (top) vs enhanced output (bottom)](plots/wiener_spectrograms.png)

### 3. Ephraim–Malah — `src/ephraim_malah.py`

The MMSE short-time spectral amplitude estimator (Ephraim & Malah, 1984).
Instead of a simple power ratio, it computes the statistically optimal
amplitude estimate assuming Gaussian speech and noise spectra, using two SNR
quantities per bin:

```
gamma = |Y|² / N                                   (a posteriori SNR)
xi    = alpha·A²ₚᵣₑᵥ/N + (1−alpha)·max(gamma−1,0)  (a priori SNR, decision-directed)
v     = xi·gamma / (1 + xi)
G     = (√pi/2)·(√v/gamma)·e^(−v/2)·[(1+v)·I₀(v/2) + v·I₁(v/2)]
```

`I₀, I₁` are modified Bessel functions, and `A_prev = G·|Y|` is the previous
frame's amplitude estimate, smoothed in with `alpha = 0.98`
("decision-directed" rule). That temporal recursion is the key difference from
the two methods above: the gain evolves smoothly over time instead of jumping
frame to frame, which strongly reduces musical noise. An a priori SNR floor of
−25 dB bounds the maximum suppression.

![Ephraim–Malah: noisy input (top) vs enhanced output (bottom)](plots/ephraim_malah_spectrograms.png)

### Comparison

Residual power measured in the VAD-detected noise-only frames of the test
signal (lower = more noise removed):

| Method | Noise-frame power | Character |
|---|---|---|
| Noisy input | 64.1 dB | — |
| Spectral subtraction | 60.2 dB | gentlest, most musical noise |
| Wiener | 58.7 dB | stronger suppression, still musical noise |
| Ephraim–Malah | 55.9 dB | strongest suppression, smooth residual |

The spectral subtraction and Wiener outputs match the book's published clips
(correlation 0.999; the small deviation is the 2048-point FFT vs. the book's
window-length FFT).

## Layout

```
data/
  Noise_attenuation.html        # full rendered chapter page (21 MB, embedded audio)
  sounds/                       # source signals: enhancement_test.wav (44.1 kHz), _16k.wav
  audio_outputs/                # all 20 audio clips embedded in the page (cellNN_clipNN.wav)
src/
  Noise_attenuation.ipynb       # original source notebook
  noise_attenuation_all_code.py # verbatim dump of every code cell
  noise_attenuation_classical.py# runnable duplicate of the chapter's classical pipeline
  spectral_subtraction.py       # standalone method 1
  wiener_filtering.py           # standalone method 2
  ephraim_malah.py              # standalone method 3
  helper_functions.py           # stft / istft / halfsinewindow / zcr (from repo)
  frontend.py, Enhancer.py      # neural demo support code (from repo)
  models/                       # pretrained weights for the three neural enhancers
plots/                          # figures (waveforms, spectrograms, noise models)
results/                        # enhanced WAVs regenerated locally
```

Local duplicate of all audio data and code from the chapter
[Noise attenuation](https://speechprocessingbook.aalto.fi/Enhancement/Noise_attenuation.html)
(source repo: [Speech-Interaction-Technology-Aalto-U/itsp](https://github.com/Speech-Interaction-Technology-Aalto-U/itsp), `Enhancement/`).
