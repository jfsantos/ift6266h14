import cPickle as pickle
import numpy as np
import theano

_mean = 0.0035805809921434142
_std = 542.48824133746177

def gen_phone(mdl, phones, noise_level):
    terr_monitor = mdl.monitor.channels['test_objective']
    terr = min(terr_monitor.val_record)

    X = theano.tensor.dmatrix('X')
    P = theano.tensor.dmatrix('P')
    y = mdl.fprop([X,P])
    predict = theano.function([X, P], y)

    # Let's start with a all zero vector, then use the prediction to populate the next sample
    duration = 3
    fs = 16000
    frame_length = mdl.input_space.components[0].dim

    x0 = np.asmatrix(np.zeros((1,duration*fs)))
    # phones = np.load('test_phones.npy')
    phone_code = np.asmatrix(np.zeros((duration*fs,3*62)))

    for pi, p in enumerate(phones):
        phone_code[pi,[p, p+62, p+2*62]] = 1 # code for 'aw'

    for k in np.arange(frame_length,duration*fs):
        frame = x0[:,k-frame_length:k]
        x0[0,k] = np.random.normal(predict(frame + np.random.normal(0, noise_level[0]*np.sqrt(terr), frame.shape), phone_code[k]), noise_level[1]*np.sqrt(terr))

    x0 = x0.T
    x0_scaled = x0*_std + _mean

    x0a = np.asarray(x0_scaled, dtype=np.int16)
    return x0a
            

if __name__ == "__main__":
    from sys import argv
    from scipy.io import wavfile
    mdl = pickle.load(open(argv[1]))
    x0a = gen_phone(mdl)
    wavfile.write(argv[2], 16000, x0a)
