import pickle
import numpy as np
import librosa
import matplotlib.pyplot as plt
import cmath
from tensorflow import keras
import tensorflow as tf


def extract_features():
	y,_ = librosa.load('rec_.wav'); sr = 22500
	y,_ = librosa.effects.trim(y)
	MFCC = librosa.feature.mfcc(y=y, sr=sr); MFCC = [np.mean(x) for x in MFCC]
	y_harmonic, y_percussive = librosa.effects.hpss(y)
	y_harmonic, y_percussive = y_harmonic.mean(), y_percussive.mean()
	C = librosa.cqt(y); C_mean = [np.mean(x) for x in C]; C_mean = [complex(x).real for x in C_mean]
	chroma = librosa.feature.chroma_cqt(C=C, sr=sr); chroma_mean = [np.mean(x) for x in chroma]
	a, b = [],[]
	for j in range(len(chroma_mean)):
		polar = cmath.polar(complex(chroma_mean[j])); a.append(polar[0]); b.append(polar[1])
	onset_envelope = librosa.onset.onset_strength(y=y, sr=sr); onsets = librosa.onset.onset_detect(onset_envelope=onset_envelope)
	onsets = onsets.shape[0]; tempo, beats = librosa.beat.beat_track(onset_envelope=onset_envelope)
	c_sync = librosa.util.sync(chroma, beats, aggregate=np.median); c_sync = [np.mean(x) for x in c_sync]
	c, d = [],[]
	for j in range(len(c_sync)):
		polar = cmath.polar(complex(c_sync[j])); c.append(polar[0]); d.append(polar[1])
	spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr); spectral_bandwidth = spectral_bandwidth.mean()
	spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]; spectral_rolloff = spectral_rolloff.mean()
	spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr); spectral_centroids = spectral_centroids.mean()
	arr = np.hstack(([], MFCC, y_harmonic, y_percussive, C_mean, a, b, onsets, tempo, beats.shape[0], c,d, spectral_bandwidth, spectral_rolloff, spectral_centroids,1,0,0,0,0))
	return y, arr

y, arr = extract_features()
emotions = ['生气', '厌恶', '恐惧', '高兴', '中性', '伤心']
with open('speech_emotion_classifier.pkl','rb') as f:
    model = pickle.load(f)
    pred = model.predict(arr.reshape((1,165)))
    print(emotions[int(np.where( pred == pred.max() )[1])])