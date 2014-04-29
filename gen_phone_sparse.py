import cPickle as pickle
import numpy as np
import theano
from sparse_coding.sparse_coding_gammatone import gammatone_matrix, erb_space

def gen_phone(mdl):
    X = theano.tensor.dmatrix('X')
    P = theano.tensor.dmatrix('P')
    y = mdl.fprop([X,P])
    predict = theano.function([X, P], y)

    resolution = 1600
    step = 64
    b = 1.019
    n_channels = 64

    D_multi = np.r_[tuple(gammatone_matrix(b, fc, resolution, step) for
                      fc in erb_space(150, 8000, n_channels))]

    phones = np.load('test_phones_1600.npy')
    X = np.asmatrix((len(phones),np.zeros(1536)))

    phone_code = np.asmatrix((len(phones),np.zeros(3*62)))

    for pi, p in enumerate(phones):
        phone_code[pi,[p[0], p[1]+62, p[2]+2*62]] = 1 # one-hot encoding

    out = np.zeros(1600 + 200*(len(phones)-1))
    step = 200
    for k in range(1,len(phones)):
        idx = range(k*step, k*step+1600)
        X[k] = predict(X[k-1], phone_code[k])
        out[idx] += np.dot(X[k], D_multi)
    
    out_scaled = np.asarray(out/max(abs(out)), dtype='float32')
    return out_scaled

if __name__ == "__main__":
    from sys import argv
    from scipy.io import wavfile
    mdl = pickle.load(open(argv[1]))
    x0a = gen_phone(mdl)
    wavfile.write(argv[2], 16000, out_scaled)
