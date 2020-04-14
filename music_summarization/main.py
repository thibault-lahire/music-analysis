import os, wave
import math

from numba import jit

import numpy as np
import scipy
import pyaudio
import matplotlib.pyplot as plt

from scipy.io.wavfile import write
from scipy.io import wavfile
import scipy.signal


def play_music(file, chunk = 1024):
    """
    Script from PyAudio doc
    """
    wf = wave.open(file, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    data = wf.readframes(chunk)

    while data:
        stream.write(data)
        data = wf.readframes(chunk )

    stream.stop_stream()
    stream.close()
    p.terminate()
    

def write_file(data, Fs, name):
    scaled = np.int16(data/np.max(np.abs(data)) * 32767)
    write(name+'.wav', Fs, scaled)
    return name+'.wav'

def listen_file(filename):
    current_path = os.getcwd()
    data_path = os.path.join(current_path, 'excerpts')
    music = os.path.join(data_path, filename)
    play_music(music)

plt.ion()

Fs, x = wavfile.read('musics/toy_music.wav')
if False: # Turn to True if x has two channels 
    x = x[:,0]

nfft=2205
noverlap=1102
window=np.hamming(nfft)
excerpt_duration = 10 # in seconds
f, t, s = scipy.signal.spectrogram(x, fs=Fs, window=window, noverlap=noverlap)

L = np.argwhere(t>excerpt_duration)[0,0]
D = s.T@s

@jit
def func(D, s):
    for i in range(D.shape[0]):
        if i%100==0:
            string = str(i) + ' elements have been processed out of ' + str(D.shape[0])
            print(string)
        for j in range(D.shape[1]):
            D[i,j] = D[i,j]/(np.linalg.norm(s.T[i])*np.linalg.norm(s.T[j]))
    return D

D = func(D, s)


plt.figure(1, figsize=(10,10))
plt.imshow(D, cmap=plt.cm.gray, interpolation = None, extent=[0,t[-1],t[-1],0])
plt.xlabel('time (seconds)')
plt.ylabel('time (seconds)')
plt.title('Similarity matrix')
#plt.suptitle('The brighter, the most similar')
plt.savefig('similarity_matrix_toy_music.png')

# Compute excerpt scores

for i in range(D.shape[0]):
    for j in range(D.shape[1]):
        if math.isnan(D[i,j]):
            D[i,j] = 0

Q = np.zeros(len(D)-L)
for i in range(Q.shape[0]):
    Q[i] = np.sum(np.sum(D[i:i+L], axis=0))
Q = Q/(D.shape[0]*L)

plt.figure(2)
plt.clf()
plt.plot(t[:-L], Q)
plt.xlabel('time (seconds)')
plt.ylabel('similarity score')
plt.title('Similarity score wrt time')
plt.savefig('scores_toy_music.png')
plt.show()

q = np.argmax(Q)

excerpt=x[int(q*(nfft-noverlap)):int((q+L)*(nfft-noverlap))]

write_file(excerpt, Fs, 'excerpts/toy_music_summary')
play_music('excerpts/toy_music_summary.wav')

