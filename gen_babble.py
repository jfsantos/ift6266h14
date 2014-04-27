import cPickle as pickle
import numpy as np
import theano
from scipy.io import wavfile
from random import randint
from pylearn2_timit.timitnext import TIMITnext

bestmdl = pickle.load(open('nextsample.pkl'))

X = theano.tensor.dmatrix('X')
y = bestmdl.fprop(X)
predict = theano.function([X], y)

# Let's start with a all zero vector, then use the prediction to populate the next sample
duration = 2
fs = 16000

x0 = np.asmatrix(np.zeros((1,duration*fs)))

def norm_data(x):
    return (x - TIMITnext._mean)/TIMITnext._std

def denorm_data(x):
    return (x*TIMITnext._std) + TIMITnext._mean

for k in np.arange(161,duration*fs):
    frame = x0[:,k-161:k-1]
    x0[0,k] = denorm_data(predict(norm_data(frame)))

x0 = x0.T

x0a = np.asarray((x0/max(abs(x0)))*2**15, dtype=np.int16)
wavfile.write('babble.wav', fs, x0a)











