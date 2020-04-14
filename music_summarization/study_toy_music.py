import os, wave

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


Fs, x = wavfile.read('musics/toy_music.wav')
nfft=2205
noverlap=1102
window=np.hamming(nfft)
excerpt_durations = [10, 20, 30, 40] # in seconds

f, t, s = scipy.signal.spectrogram(x, fs=Fs, window=window, noverlap=noverlap)

plt.figure(figsize=(10,5))

for excerpt_duration in excerpt_durations:
    L = np.argwhere(t>excerpt_duration)[0,0]
    D = s.T@s
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            D[i,j] = D[i,j]/(np.linalg.norm(s.T[i])*np.linalg.norm(s.T[j]))
    # Compute excerpt scores
    Q = np.zeros(len(D)-L)
    for i in range(Q.shape[0]):
        Q[i] = np.sum(np.sum(D[i:i+L]))
    lab = str(excerpt_duration) + ' second excerpt'
    plt.plot(t[:-L], Q/max(Q), label=lab)
plt.xlabel('time (seconds)')
plt.ylabel('similarity score')
plt.title('Similarity score wrt time for the toy music')
plt.legend()
plt.xlim(0, 100)
plt.savefig('similarity_score_toy_pb.png')



