# %% [cell 1]
# Initialization for all
from scipy.io import wavfile
import numpy as np 
import matplotlib.pyplot as plt
import IPython.display as ipd 
import scipy 
import scipy.fft 

#from helper_functions import stft, istft, halfsinewindow

# %% [cell 2]
def stft(data,fs,window_length_ms=30,window_step_ms=20,windowing_function=None):
    window_length = int(window_length_ms*fs/2000)*2
    window_step = int(window_step_ms*fs/1000)
    if windowing_function is None:
        windowing_function = np.sin(np.pi*np.arange(0.5,window_length,1)/window_length)**2
    
    total_length = len(data)
    window_count = int( (total_length-window_length)/window_step) + 1
    
    spectrum_length = int((window_length)/2)+1
    spectrogram = np.zeros((window_count,spectrum_length),dtype=complex)

    for k in range(window_count):
        starting_position = k*window_step

        data_vector = data[starting_position:(starting_position+window_length),]
        window_spectrum = scipy.fft.rfft(data_vector*windowing_function,n=window_length)

        spectrogram[k,:] = window_spectrum
        
    return spectrogram

def istft(spectrogram,fs,window_length_ms=30,window_step_ms=20,windowing_function=None):
    window_length = int(window_length_ms*fs/2000)*2
    window_step = int(window_step_ms*fs/1000)
    #if windowing_function is None:
    #    windowing_function = np.ones(window_length)
    window_count = spectrogram.shape[0]
    
    total_length = (window_count-1)*window_step + window_length
    data = np.zeros(total_length)
    
    for k in range(window_count):
        starting_position = k*window_step
        ix = np.arange(starting_position,starting_position+window_length)

        thiswin = scipy.fft.irfft(spectrogram[k,:],n=window_length)
        data[ix] = data[ix] + thiswin*windowing_function
        
    return data


def halfsinewindow(window_length):
    return np.sin(np.pi*np.arange(0.5,window_length,1)/window_length)

# %% [cell 3]
fs = 44100  # Sample rate
seconds = 5  # Duration of recording
window_length_ms=30
window_step_ms=15
window_length = int(window_length_ms*fs/2000)*2
window_step_samples = int(window_step_ms*fs/1000)

windowing_function = halfsinewindow(window_length)

filename = 'sounds/enhancement_test.wav'

# %% [cell 4]
# read from storage
fs, data = wavfile.read(filename)
data = data[:]

ipd.display(ipd.Audio(data,rate=fs))

plt.figure(figsize=[12,6])
plt.subplot(211)
t = np.arange(0,len(data),1)/fs

plt.plot(t,data)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Waveform of noisy audio')
plt.axis([0, len(data)/fs, 1.05*np.min(data), 1.05*np.max(data)])


spectrogram_matrix = stft(data,
                          fs,
                          window_length_ms=window_length_ms,
                          window_step_ms=window_step_ms,
                         windowing_function=windowing_function)
fft_length = spectrogram_matrix.shape[1]
window_count = spectrogram_matrix.shape[0]
length_in_s = window_count*window_step_ms/1000
plt.subplot(212)
plt.imshow(20*np.log10(np.abs(spectrogram_matrix[:,range(fft_length)].T)),
           origin='lower',aspect='auto',
           extent=[0, length_in_s, 0, fs/2000])
plt.axis([0, length_in_s, 0, 8])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (kHz)');
plt.title('Spectrogram of noisy audio')
plt.tight_layout()
plt.show()

# %% [cell 6]
frame_energy = np.sum(np.abs(spectrogram_matrix)**2,axis=1)
frame_energy_dB = 10*np.log10(frame_energy)
mean_energy_dB = np.mean(frame_energy_dB) # mean of energy in dB

threshold_dB = mean_energy_dB + 3. # threshold relative to mean

speech_active = frame_energy_dB > threshold_dB

# %% [cell 7]
# Reconstruct and play thresholded signal
spectrogram_thresholded = spectrogram_matrix * np.expand_dims(speech_active,axis=1)
data_thresholded = istft(spectrogram_thresholded,fs,window_length_ms=window_length_ms,window_step_ms=window_step_ms,windowing_function=windowing_function)

# Illustrate thresholding (without hysteresis)
plt.figure(figsize=[12,6])
plt.subplot(211)

t = np.arange(0,window_count,1)*window_step_samples/fs
normalized_frame_energy = frame_energy_dB - np.mean(frame_energy_dB)
plt.plot(t,normalized_frame_energy,label='Signal energy')
plt.plot(t,speech_active*10,label='Noise gate')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Noise gate')
plt.axis([0, len(data)/fs, 1.05*np.min(normalized_frame_energy), 1.05*np.max(normalized_frame_energy)])


plt.subplot(212)

plt.imshow(20*np.log10(1e-6+np.abs(spectrogram_thresholded[:,range(fft_length)].T)),
           origin='lower',aspect='auto',
           extent=[0, length_in_s, 0, fs/2000])
plt.axis([0, length_in_s, 0, 8])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (kHz)');
plt.title('Spectrogram of gated audio')
plt.tight_layout()
plt.show()
#sd.play(data_thresholded,fs)
ipd.display(ipd.Audio(data_thresholded,rate=fs))



# %% [cell 9]
hysteresis_time_ms = 300
hysteresis_time = int(hysteresis_time_ms/window_step_ms)

speech_active_hysteresis = np.zeros([window_count])
for window_ix in range(window_count):
    range_start = max(0,window_ix-hysteresis_time)
    speech_active_hysteresis[window_ix] = np.max(speech_active[range(range_start,window_ix+1)])

# %% [cell 10]
# Reconstruct and play thresholded signal
spectrogram_hysteresis = spectrogram_matrix * np.expand_dims(speech_active_hysteresis,axis=1)
data_hysteresis = istft(spectrogram_hysteresis,fs,window_length_ms=window_length_ms,window_step_ms=window_step_ms,windowing_function=windowing_function)

# Illustrate thresholding (without hysteresis)
plt.figure(figsize=[12,6])
plt.subplot(211)

t = np.arange(0,window_count,1)*window_step_samples/fs
normalized_frame_energy = frame_energy_dB - np.mean(frame_energy_dB)
plt.plot(t,normalized_frame_energy,label='Signal energy')
plt.plot(t,speech_active_hysteresis*10,label='Noise gate')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Noise gate with hysteresis')
plt.axis([0, len(data)/fs, 1.05*np.min(normalized_frame_energy), 1.05*np.max(normalized_frame_energy)])


plt.subplot(212)

plt.imshow(20*np.log10(1e-6+np.abs(spectrogram_hysteresis[:,range(fft_length)].T)),
           origin='lower',aspect='auto',
           extent=[0, length_in_s, 0, fs/2000])
plt.axis([0, length_in_s, 0, 8])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (kHz)');
plt.title('Spectrogram of gating with hysteresis')
plt.tight_layout()
plt.show()
#sd.play(data_thresholded,fs)
ipd.display(ipd.Audio(data_hysteresis,rate=fs))



# %% [cell 12]
# Fade-in and fade-out
fade_in_time_ms = 50
fade_out_time_ms = 300
fade_in_time = int(fade_in_time_ms/window_step_ms)
fade_out_time = int(fade_out_time_ms/window_step_ms)

speech_active_sloped = np.zeros([window_count])
for frame_ix in range(window_count):
    if speech_active_hysteresis[frame_ix]:
        range_start = max(0,frame_ix-fade_in_time)
        speech_active_sloped[frame_ix] = np.mean(speech_active_hysteresis[range(range_start,frame_ix+1)])
    else:
        range_start = max(0,frame_ix-fade_out_time)
        speech_active_sloped[frame_ix] = np.mean(speech_active_hysteresis[range(range_start,frame_ix+1)])

# %% [cell 13]
# Reconstruct and play sloped-thresholded signal
spectrogram_sloped = spectrogram_matrix * np.expand_dims(speech_active_sloped,axis=1)
data_sloped = istft(spectrogram_sloped,fs,window_length_ms=window_length_ms,window_step_ms=window_step_ms,windowing_function=windowing_function)

# Illustrate thresholding 
plt.figure(figsize=[12,6])
plt.subplot(211)

t = np.arange(0,window_count,1)*window_step_samples/fs
normalized_frame_energy = frame_energy_dB - np.mean(frame_energy_dB)
plt.plot(t,normalized_frame_energy,label='Signal energy')
plt.plot(t,speech_active_sloped*10,label='Noise gate')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Noise gate with sloped hysteresis')
plt.axis([0, len(data)/fs, 1.05*np.min(normalized_frame_energy), 1.05*np.max(normalized_frame_energy)])


plt.subplot(212)

plt.imshow(20*np.log10(1e-6+np.abs(spectrogram_sloped[:,range(fft_length)].T)),
           origin='lower',aspect='auto',
           extent=[0, length_in_s, 0, fs/2000])
plt.axis([0, length_in_s, 0, 8])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (kHz)');
plt.title('Spectrogram of gating with sloped hysteresis')
plt.tight_layout()
plt.show()
#sd.play(data_thresholded,fs)
ipd.display(ipd.Audio(data_sloped,rate=fs))



# %% [cell 15]
hysteresis_time_ms = 100
hysteresis_time = int(hysteresis_time_ms/window_step_ms)

fade_in_time_ms = 30
fade_out_time_ms = 60
fade_in_time = int(fade_in_time_ms/window_step_ms)
fade_out_time = int(fade_out_time_ms/window_step_ms)


# NB: This is a pedagogic, but very slow implementation since it involves multiple for-loops.
spectrogram_binwise = np.zeros(spectrogram_matrix.shape,dtype=complex)
for bin_ix in range(fft_length):
    bin_energy_dB = 10.*np.log10(np.abs(spectrogram_matrix[:,bin_ix])**2)
    mean_energy_dB = np.mean(bin_energy_dB) # mean of energy in dB
    threshold_dB = mean_energy_dB + 16. # threshold relative to mean
    speech_active = bin_energy_dB > threshold_dB
    
    speech_active_hysteresis = np.zeros_like(speech_active)
    for window_ix in range(window_count):
        range_start = max(0,window_ix-hysteresis_time)
        speech_active_hysteresis[window_ix] = np.max(speech_active[range(range_start,window_ix+1)])
        
    #speech_active_sloped = np.zeros_like(spe
    for frame_ix in range(window_count):
        if speech_active_hysteresis[frame_ix]:
            range_start = max(0,frame_ix-fade_in_time)
            speech_active_sloped[frame_ix] = np.mean(speech_active_hysteresis[range(range_start,frame_ix+1)])
        else:
            range_start = max(0,frame_ix-fade_out_time)
            speech_active_sloped[frame_ix] = np.mean(speech_active_hysteresis[range(range_start,frame_ix+1)])
            
    spectrogram_binwise[:,bin_ix] = spectrogram_matrix[:,bin_ix]*speech_active_sloped

# %% [cell 16]
# Reconstruct and play sloped-thresholded signal
data_binwise = istft(spectrogram_binwise,fs,window_length_ms=window_length_ms,window_step_ms=window_step_ms,windowing_function=windowing_function)

# Illustrate thresholding 
plt.figure(figsize=[12,6])
plt.subplot(211)
plt.imshow(20*np.log10(np.abs(spectrogram_matrix[:,range(fft_length)].T)),
           origin='lower',aspect='auto',
           extent=[0, length_in_s, 0, fs/2000])
plt.axis([0, length_in_s, 0, 8])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (kHz)');
plt.title('Original spectrogram of noisy audio')
ipd.display(ipd.Audio(data,rate=fs))

plt.subplot(212)
plt.imshow(20*np.log10(1e-6+np.abs(spectrogram_binwise[:,range(fft_length)].T)),
           origin='lower',aspect='auto',
           extent=[0, length_in_s, 0, fs/2000])
plt.axis([0, length_in_s, 0, 8])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (kHz)');
plt.title('Spectrogram of bin-wise gating with sloped hysteresis')
plt.tight_layout()
plt.show()
#sd.play(data_thresholded,fs)
ipd.display(ipd.Audio(data_binwise,rate=fs))



# %% [cell 21]
# VAD through energy thresholding
frame_energy = np.sum(np.abs(spectrogram_matrix)**2,axis=1)
frame_energy_dB = 10*np.log10(frame_energy)
mean_energy_dB = np.mean(frame_energy_dB) # mean of energy in dB

noise_threshold_dB = mean_energy_dB - 3. # threshold relative to mean

noise_active = frame_energy_dB < noise_threshold_dB

# %% [cell 22]
normalized_frame_energy = frame_energy_dB - mean_energy_dB

plt.figure(figsize=[12,4])
plt.plot(t,normalized_frame_energy,label='Frame energy')
plt.plot(t,noise_active*6,label='Noise active')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Noise detection')
plt.axis([0, len(data)/fs, 1.05*np.min(normalized_frame_energy), 1.05*np.max(normalized_frame_energy)])
plt.show()

# %% [cell 23]
# Estimate noise in noise frames
noise_frames = spectrogram_matrix[noise_active,:]
noise_estimate = np.mean(np.abs(noise_frames)**2,axis=0)

# %% [cell 24]
f = np.linspace(0,fs/1000,fft_length)
plt.plot(f,10*np.log10(1e-6+noise_estimate));
plt.xlabel('Frequency (kHz)')
plt.ylabel('Magnitude (dB)');
plt.title('Noise model');
plt.show()

# %% [cell 26]
noise_estimate_minimum = np.min(np.abs(spectrogram_matrix)**2,axis=0)

# %% [cell 27]
plt.plot(f,10*np.log10(noise_estimate),label='Mean+VAD');
plt.plot(f,10*np.log10(noise_estimate_minimum),label='Minimum');
plt.legend()
plt.xlabel('Frequency (kHz)')
plt.ylabel('Magnitude (dB)');
plt.title('Noise models');

# %% [cell 30]
energy_enhanced = np.subtract(np.abs(spectrogram_matrix)**2, np.expand_dims(noise_estimate,axis=0))
energy_enhanced *= (energy_enhanced > 0)  # threshold at zero
enhancement_gain = np.sqrt(energy_enhanced/(np.abs(spectrogram_matrix)**2))
spectrogram_enhanced = spectrogram_matrix*enhancement_gain;

# %% [cell 31]
# Reconstruct and play sloped-thresholded signal
data_enhanced = istft(spectrogram_enhanced,fs,window_length_ms=window_length_ms,window_step_ms=window_step_ms,windowing_function=windowing_function)

# Illustrate thresholding 
plt.figure(figsize=[12,6])
plt.subplot(211)
plt.imshow(20*np.log10(np.abs(spectrogram_matrix[:,range(fft_length)].T)),
           origin='lower',aspect='auto',
           extent=[0, length_in_s, 0, fs/2000])
plt.axis([0, length_in_s, 0, 8])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (kHz)');
plt.title('Original spectrogram of noisy audio')
ipd.display(ipd.Audio(data,rate=fs))

plt.subplot(212)
plt.imshow(20*np.log10(1e-6+np.abs(spectrogram_enhanced[:,range(fft_length)].T)),
           origin='lower',aspect='auto',
           extent=[0, length_in_s, 0, fs/2000])
plt.axis([0, length_in_s, 0, 8])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (kHz)');
plt.title('Spectrogram after spectral subtraction')
plt.tight_layout()
plt.show()
#sd.play(data_thresholded,fs)
ipd.display(ipd.Audio(data_enhanced,rate=fs))



# %% [cell 34]
mmse_gain = energy_enhanced/(np.abs(spectrogram_matrix)**2)
spectrogram_enhanced_mmse = spectrogram_matrix*mmse_gain

# %% [cell 35]
# Reconstruct and play enhanced signal
data_enhanced_mmse = istft(spectrogram_enhanced_mmse,fs,window_length_ms=window_length_ms,window_step_ms=window_step_ms,windowing_function=windowing_function)

# Illustrate thresholding 
plt.figure(figsize=[12,6])
plt.subplot(211)
plt.imshow(20*np.log10(np.abs(spectrogram_matrix[:,range(fft_length)].T)),
           origin='lower',aspect='auto',
           extent=[0, length_in_s, 0, fs/2000])
plt.axis([0, length_in_s, 0, 8])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (kHz)');
plt.title('Original spectrogram of noisy audio')
ipd.display(ipd.Audio(data,rate=fs))

plt.subplot(212)
plt.imshow(20*np.log10(1e-6+np.abs(spectrogram_enhanced_mmse[:,range(fft_length)].T)),
           origin='lower',aspect='auto',
           extent=[0, length_in_s, 0, fs/2000])
plt.axis([0, length_in_s, 0, 8])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (kHz)');
plt.title('Spectrogram after Wiener filtering')
plt.tight_layout()
plt.show()
#sd.play(data_thresholded,fs)
ipd.display(ipd.Audio(data_enhanced_mmse,rate=fs))



# %% [cell 38]
# Illustration of hard threshold, noise filling and soft threshold
noisefill_level = 1
x = np.linspace(-4,4,100)
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=[10,10])
ax1.plot(x,0.5*(x+np.abs(x)))
ax2.plot(x,x*0 + noisefill_level,'k:')
ax2.plot(x,noisefill_level+0.5*(x-noisefill_level+np.abs(x-noisefill_level)))
ax3.plot(x,np.log(1.+np.exp(x)))
ax1.set_title('Hard threshold\n $y=\\max(0,x)$')
ax2.set_title('Noisefill\n $y=\\max(\\epsilon,x)$')
ax3.set_title('Soft threshold\n $y=\\ln(e^x+1)$');
ax1.set_xlabel('x')
ax2.set_xlabel('x')
ax3.set_xlabel('x');
ax1.set_ylabel('y');
ax1.set_aspect('equal')
ax2.set_aspect('equal')
ax3.set_aspect('equal')
ax1.grid()
ax2.grid()
ax3.grid()
plt.show()

# %% [cell 39]
# Noise fill with min(eps,x)
noisefill_threshold_dB = -60   # dBs below average noise
noisefill_level = (10**(noisefill_threshold_dB/10))*noise_estimate  
#noisefill_level = 0.2

energy_enhanced = np.abs(spectrogram_matrix)**2 - noise_estimate

#energy_noisefill = noisefill_level + 0.5*(energy_enhanced - noisefill_level + np.abs(energy_enhanced - noisefill_level))
energy_noisefill = noisefill_level + ((energy_enhanced - noisefill_level) > 0) * (energy_enhanced - noisefill_level)
mmse_noisefill_gain = np.sqrt(energy_noisefill)/np.abs(spectrogram_matrix)
spectrogram_enhanced_noisefill = spectrogram_matrix*mmse_noisefill_gain;

# %% [cell 40]
# Reconstruct and play enhanced signal
data_enhanced_noisefill = istft(spectrogram_enhanced_noisefill,fs,window_length_ms=window_length_ms,window_step_ms=window_step_ms,windowing_function=windowing_function)

# Illustrate thresholding 
plt.figure(figsize=[12,6])
plt.subplot(211)
plt.imshow(20*np.log10(np.abs(spectrogram_matrix[:,range(fft_length)].T)),
           origin='lower',aspect='auto',
           extent=[0, length_in_s, 0, fs/2000])
plt.axis([0, length_in_s, 0, 8])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (kHz)');
plt.title('Original spectrogram of noisy audio')
ipd.display(ipd.Audio(data,rate=fs))

plt.subplot(212)
plt.imshow(20*np.log10(1e-6+np.abs(spectrogram_enhanced_noisefill[:,range(fft_length)].T)),
           origin='lower',aspect='auto',
           extent=[0, length_in_s, 0, fs/2000])
plt.axis([0, length_in_s, 0, 8])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (kHz)');
plt.title('Spectrogram after Wiener filtering')
plt.tight_layout()
plt.show()
#sd.play(data_thresholded,fs)
ipd.display(ipd.Audio(data_enhanced_noisefill,rate=fs))



# %% [cell 49]
import torch
import torchaudio

from frontend import PreProcessing, PostProcessing



class SimpleEnhancer(torch.nn.Module):
    def __init__(
        self,
        spec_size = 241,
        input_samplerate = 16000,
        n_mels = 24,
        smoothing = 0.7,
        device              = 'cpu',
    ):
        super().__init__()
        self.device       = device   
        self.spec_size    = spec_size
        self.n_mels       = n_mels
        self.eval_state   = False
        self.smoothing    = smoothing
        self.melscale_transform = torchaudio.functional.melscale_fbanks(
            self.spec_size,
            f_min       = 0,
            f_max       = input_samplerate / 2.0,
            n_mels      = n_mels,
            sample_rate = input_samplerate,
            norm        = 'slaney',
        )
        self.melscale_transform = self.melscale_transform.to(device)

        self.GRU = torch.nn.GRU(
            input_size  = n_mels,
            hidden_size = 24,
            device=device
        )
        self.dense_output = torch.nn.Sequential(
            torch.nn.Linear(
                self.GRU.hidden_size,
                self.spec_size,
                device = device),
            torch.nn.Sigmoid()
        )
        
    def eval(self):
        self.eval_state = True
        return
    def train(self):
        self.eval_state = False
        return
    
    def forward(self, input_spec: torch.Tensor) -> torch.Tensor:        
        input_features = torch.matmul(
            input_spec.abs()**2,
            self.melscale_transform)

        hidden,_ = self.GRU(input_features)
        gains = self.dense_output(hidden) 

        if self.eval_state:
            for k in range(gains.shape[-2]-1):
                gains[...,k+1,:] = (
                    (1-self.smoothing)*gains[...,k+1,:] +
                    self.smoothing    *gains[...,k,:] )
            
        estimated_spec = input_spec * gains

        return estimated_spec, gains

    def oracle_gains(self, noisy_spec: torch.Tensor, clean_spec: torch.Tensor) -> torch.Tensor:
        return (clean_spec.abs()/noisy_spec.abs().clamp(min=1e-6)).clamp(max=1)


class NoiseModelEnhancer(torch.nn.Module):
    def __init__(
        self,
        input_samplerate    = 16000,
        spec_size           = 241,
        enhancer_size       = 48,
        noise_model_size    = 24,
        n_mels              = 24,
        smoothing           = 0.2,
        device              = 'cpu'
    ):
        super().__init__()
        self.device = device
        self.spec_size = spec_size
        self.eval_state   = False
        self.smoothing    = smoothing
        self.melscale_transform = torchaudio.functional.melscale_fbanks(
            self.spec_size,
            f_min       = 0,
            f_max       = input_samplerate / 2.0,
            n_mels      = n_mels,
            sample_rate = input_samplerate,
            norm        = 'slaney',
        )
        self.melscale_transform = self.melscale_transform.to(device)

        self.noise_model = torch.nn.GRU(
            input_size = n_mels,
            hidden_size = noise_model_size,
            device=device
        )
        self.enhancer = torch.nn.GRU(
            input_size = n_mels + noise_model_size,
            hidden_size = enhancer_size,
            device=device
        )
        self.dense_output = torch.nn.Sequential(
            torch.nn.Linear(
                self.enhancer.hidden_size,
                self.spec_size,
                device=device),
            torch.nn.Sigmoid()
        )
        
    def eval(self):
        self.eval_state = True
        return
    def train(self):
        self.eval_state = False
        return


    def forward(self, input_spec: torch.Tensor) -> torch.Tensor:
        input_features = torch.matmul(
            input_spec.abs()**2,
            self.melscale_transform)

        noise_estimate,_ = self.noise_model(input_features)        
        enhancer_features = torch.cat((input_features, noise_estimate),dim=-1)

        hidden,_ = self.enhancer(enhancer_features)
        gains = self.dense_output(hidden) 

        if self.eval_state:
            for k in range(gains.shape[-2]-1):
                gains[...,k+1,:] = (
                    (1-self.smoothing)*gains[...,k+1,:] +
                    self.smoothing    *gains[...,k,:] )
            
        estimated_spec = input_spec * gains

        return estimated_spec, gains

    def oracle_gains(self, noisy_spec: torch.Tensor, clean_spec: torch.Tensor) -> torch.Tensor:
        return (clean_spec.abs()/noisy_spec.abs().clamp(min=1e-6)).clamp(max=1)

class VADNoiseModelEnhancer(torch.nn.Module):
    def __init__(
        self,
        spec_size        = 241,
        input_samplerate = 16000,
        enhancer_size    = 48,
        noise_model_size = 24,
        vad_model_size   = 24,
        n_mels           = 24,
        smoothing        = 0.1,
        device='cpu'
    ):
        super().__init__()
        self.device = device
        self.spec_size = spec_size
        self.n_mels = n_mels
        self.eval_state   = False
        self.smoothing    = smoothing        
        self.melscale_transform = torchaudio.functional.melscale_fbanks(
            self.spec_size,
            f_min       = 0,
            f_max       = input_samplerate / 2.0,
            n_mels      = n_mels,
            sample_rate = input_samplerate,
            norm        = 'slaney',
        )
        self.melscale_transform = self.melscale_transform.to(device)

        self.VAD = torch.nn.GRU(
            input_size = n_mels,
            hidden_size = vad_model_size,
            device=device
        )
        self.VAD_output = torch.nn.Linear(vad_model_size,1,device=device)
        
        self.noise_model = torch.nn.GRU(
            input_size = n_mels + vad_model_size,
            hidden_size = noise_model_size,
            device=device
        )
        
        self.enhancer = torch.nn.GRU(
            input_size = n_mels + noise_model_size + vad_model_size,
            hidden_size = enhancer_size,
            device=device
        )
        self.dense_output = torch.nn.Sequential(
            torch.nn.Linear(
                self.enhancer.hidden_size,
                self.spec_size,
                device=device),
            torch.nn.Sigmoid()
        )
        
    def eval(self):
        self.eval_state = True
        return
    def train(self):
        self.eval_state = False
        return


    def forward(self, input_spec: torch.Tensor) -> torch.Tensor:
        input_features = torch.matmul(
            input_spec.abs()**2,
            self.melscale_transform)

        vad_estimate,_ = self.VAD(input_features)
        vad_output = self.VAD_output(vad_estimate)

        noise_estimate,_ = self.noise_model(torch.cat(
            (input_features,
             vad_estimate),
            dim=-1))
        hidden,_ = self.enhancer(torch.cat(
            (input_features,
             noise_estimate, 
             vad_estimate),
            dim=-1))
        gains = self.dense_output(hidden) 

        if self.eval_state:
            for k in range(gains.shape[-2]-1):
                gains[...,k+1,:] = (
                    (1-self.smoothing)*gains[...,k+1,:] +
                    self.smoothing    *gains[...,k,:] )
            
        estimated_spec = input_spec * gains

        return estimated_spec, gains

    def oracle_gains(self, noisy_spec: torch.Tensor, clean_spec: torch.Tensor) -> torch.Tensor:
        return (clean_spec.abs()/noisy_spec.abs().clamp(min=1e-6)).clamp(max=1)



# %% [cell 50]
target_rate = 16000

# read from storage
device = 'cpu'
fs, data = wavfile.read(filename)
print(f"Input sample rate {fs} Hz")
data_torch = torch.Tensor(data[:],device=device)

preprocessor = PreProcessing(input_samplerate=fs)
postprocessor = PostProcessing(output_samplerate=fs)

enhancer_state_dict = torch.load('SimpleEnhancer_MelLoss.pt')
enhancer = SimpleEnhancer(spec_size = preprocessor.output_size,
                          input_samplerate = target_rate,
                          device=device)
enhancer.load_state_dict(enhancer_state_dict)
enhancer.eval()


# %% [cell 51]
noisy_spectrogram = preprocessor(data_torch)
noisy_audio = postprocessor(noisy_spectrogram)

enhanced_spectrogram,_ = enhancer(noisy_spectrogram)
enhanced_audio = postprocessor(enhanced_spectrogram)
enhanced_audio = enhanced_audio*noisy_audio.std()/enhanced_audio.std()

# %% [cell 52]
length_in_s = len(noisy_audio)/fs
plt.figure(figsize=(5,6))
plt.subplot(211)
plt.imshow(noisy_spectrogram.abs().log().T.numpy(),
           origin='lower',aspect='auto',
           extent=[0, length_in_s, 0, 16000/2000])
plt.axis([0, length_in_s, 0, 8])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (kHz)');
plt.title('Noisy')

plt.subplot(212)
plt.imshow(enhanced_spectrogram.abs().log().T.detach().numpy(),
           origin='lower',aspect='auto',
           extent=[0, length_in_s, 0, 16000/2000])
plt.axis([0, length_in_s, 0, 8])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (kHz)');
plt.title('Simple DNN Estimate')
plt.tight_layout()
plt.show()

import IPython
IPython.display.display(IPython.display.Audio(noisy_audio,rate=int(fs)))
IPython.display.display(IPython.display.Audio(enhanced_audio.detach().numpy(),rate=int(fs)))

# %% [cell 55]
# read from storage
fs, data = wavfile.read(filename)
data_torch = torch.Tensor(data[:],device=device)


enhancer_state_dict = torch.load('NoiseModelEnhancer_MelLoss.pt')
enhancer = NoiseModelEnhancer(spec_size = preprocessor.output_size,
                          input_samplerate = target_rate,
                          device=device)
enhancer.load_state_dict(enhancer_state_dict)
enhancer.eval()



# %% [cell 56]
noisy_spectrogram = preprocessor(data_torch)
noisy_audio = postprocessor(noisy_spectrogram)

enhanced_spectrogram,_ = enhancer(noisy_spectrogram)
enhanced_audio = postprocessor(enhanced_spectrogram)
enhanced_audio = enhanced_audio*noisy_audio.std()/enhanced_audio.std()

# %% [cell 57]
length_in_s = len(noisy_audio)/fs
plt.figure(figsize=(5,6))
plt.subplot(211)
plt.imshow(noisy_spectrogram.abs().log().T.numpy(),
           origin='lower',aspect='auto',
           extent=[0, length_in_s, 0, 16000/2000])
plt.axis([0, length_in_s, 0, 8])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (kHz)');
plt.title('Noisy')

plt.subplot(212)
plt.imshow(enhanced_spectrogram.abs().log().T.detach().numpy(),
           origin='lower',aspect='auto',
           extent=[0, length_in_s, 0, 16000/2000])
plt.axis([0, length_in_s, 0, 8])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (kHz)');
plt.title('DNN with Noise Model Estimate')
plt.tight_layout()
plt.show()

import IPython
IPython.display.display(IPython.display.Audio(noisy_audio,rate=int(fs)))
IPython.display.display(IPython.display.Audio(enhanced_audio.detach().numpy(),rate=int(fs)))

# %% [cell 60]
# read from storage
fs, data = wavfile.read(filename)
data_torch = torch.Tensor(data[:],device=device)

enhancer_state_dict = torch.load('VADNoiseModelEnhancer_MelLoss.pt')
enhancer = VADNoiseModelEnhancer(spec_size = preprocessor.output_size,
                          input_samplerate = target_rate,
                          device=device)
enhancer.load_state_dict(enhancer_state_dict)
enhancer.eval()

# %% [cell 61]
noisy_spectrogram = preprocessor(data_torch)
noisy_audio = postprocessor(noisy_spectrogram)

enhanced_spectrogram,_ = enhancer(noisy_spectrogram)
enhanced_audio = postprocessor(enhanced_spectrogram)
enhanced_audio = enhanced_audio*noisy_audio.std()/enhanced_audio.std()

# %% [cell 62]
length_in_s = len(noisy_audio)/fs
plt.figure(figsize=(5,6))
plt.subplot(211)
plt.imshow(noisy_spectrogram.abs().log().T.numpy(),
           origin='lower',aspect='auto',
           extent=[0, length_in_s, 0, 16000/2000])
plt.axis([0, length_in_s, 0, 8])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (kHz)');
plt.title('Noisy')

plt.subplot(212)
plt.imshow(enhanced_spectrogram.abs().log().T.detach().numpy(),
           origin='lower',aspect='auto',
           extent=[0, length_in_s, 0, 16000/2000])
plt.axis([0, length_in_s, 0, 8])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (kHz)');
plt.title('DNN with VAD & Noise Model Estimate')
plt.tight_layout()
plt.show()

import IPython
IPython.display.display(IPython.display.Audio(noisy_audio,rate=int(fs)))
IPython.display.display(IPython.display.Audio(enhanced_audio.detach().numpy(),rate=int(fs)))

# %% [cell 64]
from speechbrain.inference.enhancement import SpectralMaskEnhancement

enhancer = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/metricgan-plus-voicebank",
    savedir='.',
)
enhanced_audio_speechbrain = enhancer.enhance_file("sounds/enhancement_test_16k.wav")

# %% [cell 65]
fs, data = wavfile.read("sounds/enhancement_test_16k.wav")
data_torch = torch.Tensor(data[:],device=device)
preprocessor16 = PreProcessing(input_samplerate=fs)
noisy_spectrogram = preprocessor16(data_torch)
enhanced_spectrogram_speechbrain = preprocessor16(enhanced_audio_speechbrain)

length_in_s = len(data)/fs
plt.figure(figsize=(5,6))
plt.subplot(211)
plt.imshow(noisy_spectrogram.abs().log().T.numpy(),
           origin='lower',aspect='auto',
           extent=[0, length_in_s, 0, 16000/2000])
plt.axis([0, length_in_s, 0, 8])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (kHz)');
plt.title('Noisy')

plt.subplot(212)
plt.imshow(enhanced_spectrogram_speechbrain.abs().log().T.detach().numpy(),
           origin='lower',aspect='auto',
           extent=[0, length_in_s, 0, 16000/2000])
plt.axis([0, length_in_s, 0, 8])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (kHz)');
plt.title('Enhanced with SpeechBrain DNN')
plt.tight_layout()
plt.show()

import IPython
IPython.display.display(IPython.display.Audio(data,rate=int(fs)))
IPython.display.display(IPython.display.Audio(enhanced_audio_speechbrain.detach().numpy(),rate=int(fs)))

# %% [cell 69]

