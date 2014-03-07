from timit_full import TimitFullCorpusReader
import itertools
import numpy as np
from pylearn2.datasets import DenseDesignMatrix
from pylearn2.models.mlp import *
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.termination_criteria import EpochCounter
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.training_algorithms import learning_rule
from pylearn2.train import Train
from pylearn2.train_extensions import best_params
import cPickle as pickle
import theano

# Gets all utterances from <spkrid>, splits them into <framelen>
# frames with <overlap> overlaps. Returns the frames and correspondent
# phone symbols.

spkrid = 'MTCS0'

class TimitPhoneData(DenseDesignMatrix):
    def __init__(self, spkrid, phone, framelen, overlap, start, stop):
        data = TimitFullCorpusReader('/home/jfsantos/data/TIMIT/')
        # Some list comprehension/zip magic here (but it works!)
        spkrfr = [data.frames(z, 160, 159) for z in
             data.utteranceids(spkrid=spkrid)]
        fr, ph = zip(*[(x[0], x[1]) for x in spkrfr])
        fr = np.vstack(fr)*2**-15
        ph = list(itertools.chain(*ph))

        # Get all elements for which the phone is 'iy'
        iy_idx = [i for i,x in enumerate(ph) if x == 'iy']
        
        fr_iy = fr[iy_idx]

        X = fr_iy[:,0:159]
        y = np.array([fr_iy[:,159]]).T # y.ndim has to be 2

        super(TimitPhoneData,self).__init__(X=X[start:stop], y=y[start:stop])

train = TimitPhoneData(spkrid='FPLS0', phone='iy', framelen=160, overlap=159, start=0, stop=10000)
valid = TimitPhoneData(spkrid='FPLS0', phone='iy', framelen=160, overlap=159, start=10000, stop=12000)
test = TimitPhoneData(spkrid='FPLS0', phone='iy', framelen=160, overlap=159, start=12000, stop=18000)

i0 = VectorSpace(159)
s0 = Sigmoid(layer_name='h0', dim=500, sparse_init=15)
l0 = Linear(layer_name='y', dim=1, sparse_init=15)

mdl = MLP(layers=[s0, l0], nvis=159, input_space=i0)

trainer = SGD(batch_size=512, learning_rate = .01, init_momentum = .5,
              monitoring_dataset = {'train' : train, 'valid': valid,
                                    'test' : test}, termination_criterion =
              EpochCounter(max_epochs=200))

watcher = best_params.MonitorBasedSaveBest(
    channel_name='test_objective',
    save_path='nextsample_iy_FPLS0_mlp_sig_lin_watcher.pkl')
                  
experiment = Train(dataset=train,
                   model=mdl,
                   algorithm=trainer, extensions = [watcher])

experiment.main_loop()

# Now we have the best model, let's load it and use it to generate some
# samples!

bestmdl = pickle.load(open('nextsample_iy_FPLS0_mlp_sig_lin_watcher.pkl'))

X = theano.tensor.dmatrix('X')
y = bestmdl.fprop(X)
predict = theano.function([X], y)

# Let's start with a all zero vector, then use the prediction to populate the next sample
x0 = np.asmatrix(np.zeros((1,16000)))

for k in np.arange(160,16000):
    frame = x0[:,k-160:k-1]
    x0[0,k] = predict(frame)



















