import numpy as np
import sounddevice as sd
import scipy.signal as signal
import librosa
from collections import deque
import raylib as rl  # C binding for raylib

# Audio parameters
default_samplerate = 44100
block_size = 2048
n_fft = block_size
hop_length = block_size // 2

# Visualization parameters
n_bars = 64  # Number of bars in the FFT display
bar_width = None  # computed later
max_db = 80.0  # dynamic range for display
current_bar_vals = [0.0] * n_bars  # persistent bar values

# Design filters
def design_filters(fs):
    sos_hp = signal.butter(4, 100, btype='highpass', fs=fs, output='sos')
    sos_lp = signal.butter(4, 15000, btype='lowpass', fs=fs, output='sos')
    freqs = [50, 100, 150]
    q = 30.0
    sos_notch_list = [signal.tf2sos(*signal.iirnotch(f, Q=q, fs=fs)) for f in freqs]
    sos_notch = np.vstack(sos_notch_list)
    return sos_hp, sos_notch, sos_lp

# Estimate noise threshold during calibration
def estimate_noise_threshold(y_init, n_fft, hop_length, n_std_thresh=1.5):
    D = librosa.stft(y_init, n_fft=n_fft, hop_length=hop_length)
    noise_mean = np.mean(np.abs(D), axis=1, keepdims=True)
    return noise_mean * n_std_thresh

# Spectral noise reduction using librosa
def reduce_noise_librosa(y, noise_thresh, n_fft, hop_length):
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S, phase = np.abs(D), np.angle(D)
    mask = S >= noise_thresh
    S_clean = S * mask
    D_clean = S_clean * np.exp(1j * phase)
    return librosa.istft(D_clean, hop_length=hop_length)

# Simple compressor
def compress(audio, threshold=-20.0, ratio=2.0, makeup_gain=1.0, eps=1e-6):
    rms = np.sqrt(np.mean(audio**2) + eps)
    db = 20 * np.log10(rms + eps)
    if db > threshold:
        gain_db = threshold + (db - threshold) / ratio - db
    else:
        gain_db = 0
    return audio * (10 ** (gain_db / 20.0)) * makeup_gain

# Audio processing callback
def audio_callback(indata, outdata, frames, time, status):
    global sos_hp, sos_notch, sos_lp, noise_thresh, current_bar_vals
    if status and not getattr(audio_callback, 'warned', False):
        print(status)
        audio_callback.warned = True
    audio = indata[:, 0]
    # Filtering
    y = signal.sosfilt(sos_hp, audio)
    y = signal.sosfilt(sos_notch, y)
    y = signal.sosfilt(sos_lp, y)
    # Noise reduction & compression
    y = reduce_noise_librosa(y, noise_thresh, n_fft, hop_length)
    y = compress(y)
    # Mute output to avoid echo
    outdata[:] = 0
    # Compute FFT magnitudes in dB
    fft = np.abs(np.fft.rfft(y, n=n_fft))
    fft_db = 20 * np.log10(fft + 1e-6)
    # Normalize 0 to max_db
    fft_db_norm = np.clip(fft_db + max_db, 0, max_db) / max_db
    # Aggregate into bars
    bins = np.array_split(fft_db_norm, n_bars)
    current_bar_vals = [float(np.mean(b)) for b in bins]

# Visualization with raylib: bar-style FFT display
def run_visualization():
    global bar_width, current_bar_vals
    width, height = 800, 600
    bar_width = width // n_bars
    rl.InitWindow(width, height, b"Real-Time Audio FFT")

    while not rl.WindowShouldClose():
        rl.BeginDrawing()
        rl.ClearBackground((0, 0, 0, 255))
        # Draw each bar based on latest current_bar_vals
        for i, v in enumerate(current_bar_vals):
            bar_height = int(v * height)
            x = i * bar_width
            intensity = int(v * 255)
            # Color gradient: blueish
            color = (intensity, intensity, 255)
            rl.DrawRectangle(x, height - bar_height, bar_width - 2, bar_height, (*color, 255))
        rl.EndDrawing()

    rl.CloseWindow()

if __name__ == "__main__":
    print("Calibrating noise floor... (silence)")
    y_init = sd.rec(int(default_samplerate * 1.0), samplerate=default_samplerate, channels=1)
    sd.wait()
    noise_thresh = estimate_noise_threshold(y_init.flatten(), n_fft, hop_length)

    sos_hp, sos_notch, sos_lp = design_filters(default_samplerate)
    # Start audio stream
    with sd.Stream(channels=1, samplerate=default_samplerate,
                   blocksize=block_size, latency='high', callback=audio_callback):
        run_visualization()
