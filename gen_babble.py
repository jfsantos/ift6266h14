import cPickle as pickle
import numpy as np
import theano
from scipy.io import wavfile

bestmdl = pickle.load(open('nextsample_mlp_sig_lin.pkl'))

X = theano.tensor.dmatrix('X')
y = bestmdl.fprop(X)
predict = theano.function([X], y)

# Let's start with a all zero vector, then use the prediction to populate the next sample
duration = 2
fs = 16000

x0 = np.asmatrix(np.zeros((1,duration*fs)))
x0[0,0:159] = 0.3*np.sin(125*2*np.pi*np.linspace(0,2,159))

for k in np.arange(160,duration*fs):
    frame = x0[:,k-160:k-1]
    x0[0,k] = predict(frame)

x0 = x0.T

x0a = np.asarray((x0/max(abs(x0)))*2**15, dtype=np.int16)
wavfile.write('babble.wav', fs, x0a)

