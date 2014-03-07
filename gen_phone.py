import cPickle as pickle
import numpy as np
import theano
from scipy.io import wavfile

_mean = 0.0035805809921434142
_std = 542.48824133746177

bestmdl = pickle.load(open('test_phone_best.pkl'))

terr_monitor = bestmdl.monitor.channels['train_objective']
terr = min(terr_monitor.val_record)

X = theano.tensor.dmatrix('X')
P = theano.tensor.dmatrix('P')
y = bestmdl.fprop([X,P])
predict = theano.function([X, P], y)

# Let's start with a all zero vector, then use the prediction to populate the next sample
duration = 5
fs = 16000

x0 = np.asmatrix(np.zeros((1,duration*fs)))
x0[0,0:159] = np.random.normal(0, np.sqrt(terr), size=(1,159))
phone = [15, 34, 10, 51, 48]
phone_code = np.asmatrix(np.zeros((duration*fs,3*62)))

for pi, p in enumerate(phone):
    phone_code[pi*fs:(pi+1)*fs,[p, p+62, p+2*62]] = 1 # code for 'aw'


for k in np.arange(160,duration*fs):
    frame = x0[:,k-160:k]
    x0[0,k] = np.random.normal(predict(frame, phone_code[k]), np.sqrt(terr))

x0 = x0.T

x0a = np.asarray(x0*_std + _mean, dtype=np.int16)
wavfile.write('aw.wav', fs, x0a)

