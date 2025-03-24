import numpy as np
from tinytag import TinyTag as tt
import os
import array
import soundfile as sf
from multiprocessing import Pool

def process_song(song):
    y, sr = sf.read(song[0])

    song_spectogram = []

    L = 10000
    for i in range(0, len(y), L):
        # end = i+L if i+L < len(y) else len(y)
        Y = np.fft.fft(y[i:i+L, 0])
        N = len(Y)
        n = np.arange(N)
        T = N/sr
        freq = n/T
        song_spectogram.append(np.abs(Y)[N//2:N])

    song_spectogram = np.array(song_spectogram[:-1]).T

    np.save(f"song_data/{os.path.splitext(os.path.basename(song[0]))[0]}.{song[1]}", song_spectogram)

if __name__ == '__main__':
    files = [f for f in os.listdir("/Users/m1cha1s/Music/MyMusic/") if f.endswith('.mp3')]

    songs = []
    tags = set()
    for file in files:
        path = "/Users/m1cha1s/Music/MyMusic/"+file
        tag = tt.get(path)
        if tag.genre:
            songs.append((path, tag.genre))
            tags.add(tag.genre)

    with Pool(12) as p:
        p.map(process_song, songs)