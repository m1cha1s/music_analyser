import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
file_path = "C:\\Users\\szymo\\OneDrive\\Pulpit\\dsp project\\Data\\genres_original\\blues\\blues.00000.wav"

if os.path.exists(file_path):
    y, sr = librosa.load(file_path, sr=22050)
else:
    print("Chuj ci w dupe")

# Compute mel spectrogram using keyword arguments for clarity.
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
# Convert the power spectrogram to decibel (dB) scale.
S_dB = librosa.power_to_db(S, ref=np.max)
plt.figure(figsize=(6,4))
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format="%+2.f dB")
plt.title("Mel Spectrogram")
plt.show()