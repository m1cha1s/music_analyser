import numpy as np
import sounddevice as sd
from scipy.signal import firwin, lfilter, spectrogram
import matplotlib.pyplot as plt

# Sampling and recording parameters
fs = 44100            # Sampling frequency (Hz)
duration = 30          # Duration of recording (seconds)

# Record audio from the default microphone
print("Recording...")
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
sd.wait()           # Wait until the recording is finished
audio = audio.flatten()  # Flatten the array to 1D in case it's 2D
print("Recording complete.")

# Design an FIR filter
# For example, creating a lowpass FIR filter with cutoff frequency of 1000 Hz
numtaps = 101         # Number of taps in the FIR filter (filter order + 1)
cutoff = 1000         # Cutoff frequency in Hz
# Normalization factor: cutoff frequency / (Nyquist frequency)
fir_coeff = firwin(numtaps, cutoff / (0.5 * fs))

# Apply the FIR filter to the recorded audio data
filtered_audio = lfilter(fir_coeff, 1.0, audio)

# Compute the spectrogram of the filtered audio
f, t, Sxx = spectrogram(filtered_audio, fs)

# Plotting the spectrogram
plt.figure(figsize=(10, 6))
# Use a logarithmic scale for the power spectrum for improved visualization (in dB)
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Spectrogram of the Filtered Audio')
cbar = plt.colorbar()
cbar.set_label('Intensity [dB]')

# Save the spectrogram to a file
#plt.savefig('spectrogram.png', dpi=300)
plt.show()

print("Spectrogram saved as 'spectrogram.png'.")
