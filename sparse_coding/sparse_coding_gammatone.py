"""
===========================================
Sparse coding with a precomputed dictionary
===========================================

This is based on the example "Sparse coding with a precomputed dictionary" from scikit-learn. I've replaced the dictionary of Ricker wavelets by a dictionary with gammatones.

"""
print(__doc__)

import numpy as np
import matplotlib.pylab as pl

from sklearn.decomposition import SparseCoder

def gammatone_function(resolution, fc, center, fs=16000, l=4,
                       b=1.019):
    t = np.linspace(0, resolution-(center+1), resolution-center)/fs
    g = np.zeros((resolution,))
    g[center:] = t**(l-1) * np.exp(-2*np.pi*b*erb(fc)*t)*np.cos(2*np.pi*fc*t)
    return g

def gammatone_matrix(b, fc, resolution, step):
    """Dictionary of Ricker (mexican hat) wavelets"""
    centers = np.arange(0, resolution - step, step)
    D = np.empty((len(centers), resolution))
    for i, center in enumerate(centers):
        D[i] = gammatone_function(resolution, fc, center, b=b)
    D /= np.sqrt(np.sum(D ** 2, axis=1))[:, np.newaxis]
    return D

def erb(f):
    return 24.7+0.108*f

def ricker_function(resolution, center, width):
    """Discrete sub-sampled Ricker (mexican hat) wavelet"""
    x = np.linspace(0, resolution - 1, resolution)
    x = ((2 / ((np.sqrt(3 * width) * np.pi ** 1 / 4)))
         * (1 - ((x - center) ** 2 / width ** 2))
         * np.exp((-(x - center) ** 2) / (2 * width ** 2)))
    return x


def ricker_matrix(width, resolution, n_components):
    """Dictionary of Ricker (mexican hat) wavelets"""
    centers = np.linspace(0, resolution - 1, n_components)
    D = np.empty((n_components, resolution))
    for i, center in enumerate(centers):
        D[i] = ricker_function(resolution, center, width)
    D /= np.sqrt(np.sum(D ** 2, axis=1))[:, np.newaxis]
    return D

def erb_space(low_freq, high_freq, num_channels, EarQ = 9.26449, minBW = 24.7, order = 1):
    return -(EarQ*minBW) + np.exp(np.arange(1,num_channels+1)*(-np.log(high_freq + EarQ*minBW) + np.log(low_freq + EarQ*minBW))/num_channels) * (high_freq + EarQ*minBW)

if __name__ == '__main__':
    from scipy.io import wavfile
    from scikits.talkbox import segment_axis
    resolution = 2048
    step = 32
    b = 1.019
    n_channels = 50
    
    # Compute a multiscale dictionary
    
    D_multi = np.r_[tuple(gammatone_matrix(b, fc, resolution, step) for
                          fc in erb_space(150, 8000, n_channels))]

    # Load test signal
    fs, y = wavfile.read('/home/jfsantos/data/TIMIT_orig/TRAIN/DR1/FCJF0/SA1.WAV')
    y = y / 2.0**15
    Y = segment_axis(y, resolution, overlap=resolution/2, end='pad')
    Y = np.hanning(resolution) * Y

    # segments should be windowed and overlap
    
    coder = SparseCoder(dictionary=D_multi, transform_n_nonzero_coefs=200, transform_alpha=None, transform_algorithm='omp')
    X = coder.transform(Y)
    density = len(np.flatnonzero(X))
    out= np.zeros((np.ceil(len(y)/resolution)+1)*resolution)
    for k in range(0, len(X)):
        idx = range(k*resolution/2,k*resolution/2 + resolution)
        out[idx] += np.dot(X[k], D_multi)
    squared_error = np.sum((y - out[0:len(y)]) ** 2)
    wavfile.write('reconst_win.wav', fs, np.asarray(out, dtype=np.float32))
