from timit_dataset import TimitFrameData
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

train = TimitFrameData('/home/jfsantos/data/TIMIT/', framelen=160, overlap=159, start=0, stop=10000)
valid = TimitFrameData('/home/jfsantos/data/TIMIT/', framelen=160, overlap=159, start=10000, stop=12000)
test = TimitFrameData('/home/jfsantos/data/TIMIT/', framelen=160, overlap=159, start=12000, stop=18000)

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
