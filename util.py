import numpy as np
import scipy.signal as ss
from scipy.io import wavfile


def wave_read(file_name):
    sampling_rate, sig = wavfile.read(file_name)
    return sig


def stft(x, fftSize, frameShift):
    ''' x: time domain signal with shape (..., nSamples, nMic)
        X: STFT result with shape (..., nMic, nFreq, nTime) '''
    x = np.moveaxis(x, -1, -2)
    _, _, X = ss.stft(x, nperseg=fftSize, noverlap=fftSize-frameShift)
    return X


def istft(Y, fftSize, frameShift):
    ''' Y: STFT components with shape (..., nMic, nFreq, nTime) 
        y: time domain result with shape (..., nSamples, nMic) '''
    noverlap = fftSize-frameShift
    _, y = ss.istft(Y, nperseg=fftSize, noverlap=noverlap)
    y = np.moveaxis(y, -1, -2)
    return y


def wave_out(y, name, rate=8000):
    ''' Write .wav file
      y: array with shape (nSamples, nMic) or (nSig, nSamples, nMic)
      name: file name suffix
    '''
    if y.ndim == 2:
        wavfile.write(name + '.wav', rate, y)
    elif y.ndim == 3:
        nSig = y.shape[0]
        for s in range(nSig):
            wavfile.write(name + str(s + 1) + '.wav', rate, y[s])
    else:
        raise Exception('Unknown shape of y')
    return
