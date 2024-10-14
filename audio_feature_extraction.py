# -*- coding: utf-8 -*-
"""Audio Feature Extraction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VtIItWmGcB6OqBr6ujOt5ToUDe5XS2Yf

## Speech Emotion Recognition System

##### **CREMA-D** Dataset : (! git clone https://github.com/CheyneyComputerScience/CREMA-D.git)
##### **Emotions** : Anger (ANG),  Disgust (DIS),  Fear (FEA),  Happy/Joy (HAP),  Neutral (NEU),  Sad (SAD).
##### **Emotion Levels** : Low (LO),  Medium (MD),  High (HI),  Unspecified (XX).
##### **Naming of files** : Actor id_Sentence_Emotion_Level.wav

#### **Imports** :
"""

! git clone https://github.com/CheyneyComputerScience/CREMA-D.git

! pip3 install librosa mir_eval

import matplotlib.pyplot as plt
import librosa
import librosa.util, librosa.display
import numpy as np
import mir_eval
import scipy
import seaborn as sns
from IPython.display import Audio

"""#### **Loading sample files from the dataset** :"""

src='/content/CREMA-D/AudioWAV'
y1, sr1 = librosa.load(src+'/1001_DFA_ANG_XX.wav')
y2, sr2 = librosa.load(src+'/1001_DFA_DIS_XX.wav')
y3, sr3 = librosa.load(src+'/1001_DFA_FEA_XX.wav')
y4, sr4 = librosa.load(src+'/1001_DFA_HAP_XX.wav')
y5, sr5 = librosa.load(src+'/1001_DFA_NEU_XX.wav')
y6, sr6 = librosa.load(src+'/1001_DFA_SAD_XX.wav')

Audio(data=y5, rate=sr5) # Neutral

Audio(data=y1, rate=sr1) # Angry

"""#### **Visualizing one audio file for each emotion** :"""

emotions = ['Angry', 'Disgust', 'Fear', 'Happy/Joy', 'Neutral', 'Sad']

fig, axes = plt.subplots(1,6, figsize=(15,3))
for i in range(1,7):
    librosa.display.waveshow(locals()['y'+str(i)], sr=22500, ax=axes[i-1])
    axes[i-1].set_title(emotions[i-1]);

"""#### **Plotting frequency domain spectrograms** :"""

fig, axes = plt.subplots(1,6, figsize=(15,3))

for i in range(1,7):
    y = locals()['y'+str(i)]
    
    # Short-time Fourier transform (STFT) = gives Frequency domain series - freq vs time - spectrogram.
    y_stft = np.abs(librosa.stft(y))
    y_stft = librosa.amplitude_to_db(y_stft, ref=np.max) # Convert Hz to DB scale.

    librosa.display.specshow(y_stft, x_axis='time', y_axis='log', ax=axes[i-1])
    axes[i-1].set_title(emotions[i-1]);

"""#### **Plotting Mel Spectrograms** :"""

fig, axes = plt.subplots(1,6, figsize=(15,3))

for i in range(1,7):
    y = locals()['y'+str(i)]
    
    # Mel Spectrogram = converts frequencies to mel scale, interpretable by humans.
    y_mel = librosa.feature.melspectrogram(y=y, sr=22500)
    y_mel_db = librosa.amplitude_to_db(y_mel, ref=np.max) # Mel Scale to DB.
    
    librosa.display.specshow(y_mel_db, x_axis='time', y_axis='log', ax=axes[i-1]);
    axes[i-1].set_title(emotions[i-1]);

"""#### **Creation of empty dataframe to store audio features** :"""

cols = np.hstack((['actor','sentence', 'emotion','level'],['mfcc'+str(i) for i in range(20)], 'y_harmonic', 'y_percussive', ['C'+str(i) for i in range(84)], ['chroma'+str(i)+"a" for i in range(12)], ['chroma'+str(i)+"b" for i in range(12)], 'onsets', 'tempo', 'beats', ['c_sync'+str(i)+"a" for i in range(12)], ['c_sync'+str(i)+"b" for i in range(12)], 'spectral_bandwidth', 'spectral_rolloff', 'spectral_centroids'))

import pandas as pd
import cmath
df = pd.DataFrame(index=[i in range(7442)], columns = cols)

df.shape

"""#### **In a loop, extracting features of all 7442 audio files** :"""

import os, glob
for i,filename in enumerate(glob.glob(os.path.join(src, '*.wav'))):
  with open(os.path.join(os.getcwd(), filename), 'r') as f:
    actor, sentence, emotion, level = filename[26:len(filename)-4].split('_')

    y,_ = librosa.load(filename)
    sr = 22500
    # Trim = Remove leading and trailing silence.
    y,_ = librosa.effects.trim(y)

    # MFCC = compressible representations of the Log Mel Spectrogram. 
    MFCC = librosa.feature.mfcc(y=y, sr=sr)
    MFCC = [np.mean(x) for x in MFCC]

    # Harmonic = sound we perceive as melodies and chords.
    # Percussive = sound which is noise-like : eg=drums.
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    y_harmonic, y_percussive = y_harmonic.mean(), y_percussive.mean()

    # CQT = computes the constant-Q transform of an audio signal - Similar to Fourier Transform.
    C = librosa.cqt(y)
    C_mean = [np.mean(x) for x in C]
    C_mean = [complex(x).real for x in C_mean]

    # Chroma = quality of a specific tone, bins the audio into 12 tones/notes - CC#DD#EFF#GG#AA#B.
    chroma = librosa.feature.chroma_cqt(C=C, sr=sr)
    chroma_mean = [np.mean(x) for x in chroma]
    a, b = [],[]
    for j in range(len(chroma_mean)):
      polar = cmath.polar(complex(chroma_mean[j]))
      a.append(polar[0])
      b.append(polar[1])

    # Onset = the beginning of a musical note, where amplitude rises from zero to an initial peak = event.
    onset_envelope = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_envelope)
    onsets = onsets.shape[0]

    # Tempo = Speed of beats in bpm.
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_envelope)

    # Sync = temporal feature - shows repitition of structure.
    c_sync = librosa.util.sync(chroma, beats, aggregate=np.median)
    c_sync = [np.mean(x) for x in c_sync]
    c, d = [],[]
    for j in range(len(c_sync)):
      polar = cmath.polar(complex(c_sync[j]))
      c.append(polar[0])
      d.append(polar[1])

    # Spectral Bandwidth = difference between the upper and lower frequencies.
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_bandwidth = spectral_bandwidth.mean()

    # Spectral Rolloff = frequency below which a specified percentage of the total spectral energy(e.g. 85 %) lies.
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_rolloff = spectral_rolloff.mean()

    # Spectral Centroids = indicates where the center of mass of the spectrum is.
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroids = spectral_centroids.mean()

    arr = np.hstack(([], actor, sentence, emotion, level, MFCC, y_harmonic, y_percussive, C_mean, a, b, onsets, tempo, beats.shape[0], c,d, spectral_bandwidth, spectral_rolloff, spectral_centroids))

    df.loc[i] = arr
    print(i, end='\r')

"""#### **Data extracted from the audio files** :"""

df.tail()

"""#### **Storing the extracted dataset as a CSV file** :"""

df.to_csv('crema.csv', index=False)

