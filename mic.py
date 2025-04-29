import time
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
record_duration = 60  # seconds

target_frames = 130  # fixed time frames for mel-spectrogram
n_mels = 128         # mel bands

# Visualization parameters
n_bars = 64  # Number of bars in the FFT display
bar_width = None
max_db = 80.0  # dynamic range for display
current_bar_vals = [0.0] * n_bars  # persistent bar values

audio_buffer = deque()  # store processed audio blocks for mel-spectrogram

# Design filters
def design_filters(fs):
    sos_hp = signal.butter(4, 100, btype='highpass', fs=fs, output='sos')
    sos_lp = signal.butter(4, 15000, btype='lowpass', fs=fs, output='sos')
    freqs = [50, 100, 150]
    q = 30.0
    sos_notch = np.vstack([signal.tf2sos(*signal.iirnotch(f, Q=q, fs=fs)) for f in freqs])
    return sos_hp, sos_notch, sos_lp

# Estimate noise threshold
def estimate_noise_threshold(y_init, n_fft, hop_length, n_std_thresh=1.5):
    D = librosa.stft(y_init, n_fft=n_fft, hop_length=hop_length)
    noise_mean = np.mean(np.abs(D), axis=1, keepdims=True)
    return noise_mean * n_std_thresh

# Spectral noise reduction
def reduce_noise_librosa(y, noise_thresh, n_fft, hop_length):
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S, phase = np.abs(D), np.angle(D)
    mask = S >= noise_thresh
    D_clean = (S * mask) * np.exp(1j * phase)
    return librosa.istft(D_clean, hop_length=hop_length)

# Simple compressor
def compress(audio, threshold=-20.0, ratio=2.0, makeup_gain=1.0, eps=1e-6):
    rms = np.sqrt(np.mean(audio**2) + eps)
    db = 20 * np.log10(rms + eps)
    gain_db = (threshold + (db - threshold) / ratio - db) if db > threshold else 0
    return audio * (10 ** (gain_db / 20.0)) * makeup_gain

# Audio processing callback
def audio_callback(indata, outdata, frames, time, status):
    global sos_hp, sos_notch, sos_lp, noise_thresh, current_bar_vals, audio_buffer
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
    # Mute output
    outdata[:] = 0
    # Buffer for mel-spectrogram
    audio_buffer.append(y)
    # Prepare FFT bars
    fft = np.abs(np.fft.rfft(y, n=n_fft))
    fft_db = 20 * np.log10(fft + 1e-6)
    fft_db_norm = np.clip(fft_db + max_db, 0, max_db) / max_db
    bins = np.array_split(fft_db_norm, n_bars)
    current_bar_vals = [float(np.mean(b)) for b in bins]

# Visualization: bar FFT and timer
def run_visualization():
    global bar_width, current_bar_vals
    width, height = 800, 600
    bar_width = width // n_bars
    rl.InitWindow(width, height, b"Real-Time Audio FFT")
    start_time = time.time()

    while not rl.WindowShouldClose() and (time.time() - start_time) < record_duration:
        rl.BeginDrawing()
        rl.ClearBackground((0, 0, 0, 255))
        # Draw FFT bars
        for i, v in enumerate(current_bar_vals):
            bar_height = int(v * height)
            x = i * bar_width
            intensity = int(v * 255)
            rl.DrawRectangle(x, height - bar_height, bar_width - 2, bar_height, (intensity, intensity, 255, 255))
        # Draw timer text
        elapsed = time.time() - start_time
        rl.DrawText(f"Time: {elapsed:.1f}s".encode('utf-8'), 10, 10, 20, (255, 255, 255, 255))
        rl.EndDrawing()

    rl.CloseWindow()

if __name__ == "__main__":
    # Calibrate noise
    print("Calibrating noise floor... (silence)")
    y_init = sd.rec(int(default_samplerate * 1.0), samplerate=default_samplerate, channels=1)
    sd.wait()
    noise_thresh = estimate_noise_threshold(y_init.flatten(), n_fft, hop_length)
    # Setup filters
    sos_hp, sos_notch, sos_lp = design_filters(default_samplerate)
    # Start audio & visualization
    with sd.Stream(channels=1, samplerate=default_samplerate,
                   blocksize=block_size, latency='high', callback=audio_callback):
        run_visualization()

    # After recording: mel-spectrogram saving
    raw_audio = np.concatenate(list(audio_buffer), axis=0)
    S = librosa.feature.melspectrogram(y=raw_audio, sr=default_samplerate,
                                       n_mels=n_mels, n_fft=2048, hop_length=512)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_dB_fixed = librosa.util.fix_length(S_dB, size=target_frames, axis=1)
    np.save("samples.npy", S_dB_fixed)
    print(f"Saved mel-spectrogram: {S_dB_fixed.shape} -> samples.npy")
    # -- Visual check --
    # Load and plot the saved spectrogram to verify
    try:
        import matplotlib.pyplot as plt
        import librosa.display
        spec = np.load("samples.npy")
        plt.figure(figsize=(8, 4))
        librosa.display.specshow(spec, sr=default_samplerate, hop_length=512,
                                 x_axis='time', y_axis='mel', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Recorded Mel-Spectrogram')
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("matplotlib or librosa.display not available for plotting.")
    print(f"Saved mel-spectrogram: {S_dB_fixed.shape} -> samples.npy")
