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

class TimitFrameData(DenseDesignMatrix):
    def __init__(self, spkrid, framelen, overlap, start, stop):
        data = TimitFullCorpusReader('/home/jfsantos/data/TIMIT/')
        # Some list comprehension/zip magic here (but it works!)
        spkrfr = [data.frames(z, framelen, overlap) for z in
             data.utteranceids(spkrid=spkrid)]
        fr, ph = zip(*[(x[0], x[1]) for x in spkrfr])
        fr = np.vstack(fr)*2**-15
        ph = list(itertools.chain(*ph))

        X = fr[:,0:framelen-1]
        y = np.array([fr[:,framelen]]).T # y.ndim has to be 2

        super(TimitFrameData,self).__init__(X=X[start:stop], y=y[start:stop])

train = TimitFrameData(spkrid=spkrid, framelen=160, overlap=159, start=0, stop=10000)
valid = TimitFrameData(spkrid=spkrid, framelen=160, overlap=159, start=10000, stop=12000)
test = TimitFrameData(spkrid=spkrid, framelen=160, overlap=159, start=12000, stop=18000)

i0 = VectorSpace(159)
s0 = Sigmoid(layer_name='h0', dim=500, sparse_init=150)
s1 = Sigmoid(layer_name='h1', dim=500, sparse_init=150)
l0 = Linear(layer_name='y', dim=1, sparse_init=150)

mdl = MLP(layers=[s0, s1, l0], nvis=159, input_space=i0)

trainer = SGD(batch_size=512, learning_rate = .01, init_momentum = .5,
              monitoring_dataset = {'train' : train, 'valid': valid,
                                    'test' : test}, termination_criterion =
              EpochCounter(max_epochs=200))

watcher = best_params.MonitorBasedSaveBest(
    channel_name='test_objective',
    save_path='nextsample_mlp_sig_lin.pkl')
                  
experiment = Train(dataset=train,
                   model=mdl,
                   algorithm=trainer, extensions = [watcher])

experiment.main_loop()
