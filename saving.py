#!/usr/bin/env python3

import os
import numpy as np
import librosa
from datasets import load_dataset
from multiprocessing import Pool

dataset = load_dataset("marsyas/gtzan", trust_remote_code=True)
genre_count = max(dataset['train']['genre'])+1
# print(genre_count, [0]*10)


target_frames = 130

def process(sample):
    samples = sample['audio']
    y = samples['array']
    sr = samples['sampling_rate']
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_dB_fixed = librosa.util.fix_length(S_dB, size=target_frames, axis=1)
    # g = np.zeros(genre_count)
    # g[sample['genre']] = 1

    # print(S_dB_fixed.shape, g.shape)
    # print(S_dB_fixed, g)

    return S_dB_fixed

def genre_dist(x):
    g = np.zeros(genre_count)
    g[x] = 1
    return g

if __name__ == '__main__':
    with Pool(16) as p:
        res = p.map(process, dataset['train'])
        genre = [genre_dist(x['genre']) for x in dataset['train']]
        np.save("samples", res)
        np.save("genres", genre)
