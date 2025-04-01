import os
import numpy as np
import librosa

# Define the root dataset path
dataset_path = "C:\\Users\\szymo\\OneDrive\\Pulpit\\dsp project\\Data\\genres_original"

# List of genre names
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Initialize lists to store spectrograms and corresponding labels
spectrograms = []
labels = []

# Set target number of frames for each spectrogram (adjust based on your dataset)
target_frames = 130

# Loop through each genre folder
for genre in genres:
    genre_folder = os.path.join(dataset_path, genre)
    print(f"Processing genre: {genre}")
    
    # Process 100 files per genre: e.g., "blues.00000.wav" to "blues.00099.wav"
    for i in range(100):
        file_name = f"{genre}.{i:05d}.wav"  # Formats number as 00000, 00001, ..., 00099
        file_path = os.path.join(genre_folder, file_name)
        
        if os.path.exists(file_path):
            try:
                # Load audio file with a fixed sample rate
                y, sr = librosa.load(file_path, sr=22050)
                # Compute the mel spectrogram
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
                # Convert the power spectrogram to decibel (log) scale
                S_dB = librosa.power_to_db(S, ref=np.max)
                # Pad or trim the spectrogram to have the same number of time frames
                S_dB_fixed = librosa.util.fix_length(S_dB, size=target_frames, axis=1)
                # Append the spectrogram and label to the lists
                spectrograms.append(S_dB_fixed)
                labels.append(genre)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")

# Convert lists into numpy arrays for training
spectrograms = np.array(spectrograms)
labels = np.array(labels)

print("Preprocessing complete.")
print("Spectrograms shape:", spectrograms.shape)
print("Labels shape:", labels.shape)
