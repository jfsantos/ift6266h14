from pylearn2_timit.timitlpc import TIMITlpc
import itertools
import numpy as np
from pylearn2.models.mlp import *
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.termination_criteria import EpochCounter
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.training_algorithms import learning_rule
from pylearn2.train import Train
from pylearn2.train_extensions import best_params
from pylearn2.costs.cost import SumOfCosts
from pylearn2.costs.mlp import *
from pylearn2.space import *
import cPickle as pickle
import theano
from sys import argv

frame_len = 160
overlap = 0

train = TIMITlpc("train", frame_len, overlap, start=0, stop=100)
valid = TIMITlpc("valid", frame_len, overlap, start=0, stop=10)
test = TIMITlpc("test", frame_len, overlap, start=0, stop=50)

formatter = OneHotFormatter(max_labels=62)
f = lambda x: formatter.format(np.asarray(x, dtype=int), mode='merge')
iter_convert = [None, f]

train._iter_convert = iter_convert
valid._iter_convert = iter_convert
test._iter_convert = iter_convert

# Model used to predict the next sample. This model is never trained,
# as it uses parameters predicted by the next model.

i0 = VectorSpace(3*62)
s0 = Sigmoid(layer_name='h0', dim=500, sparse_init=150)
s1 = Sigmoid(layer_name='h1', dim=250, sparse_init=75)
s2 = Sigmoid(layer_name='h2', dim=100, sparse_init=15)
l0 = Linear(layer_name='y', dim=10, sparse_init=1)

mdl = MLP(layers=[s0, s1, s2, l0], nvis=3*62, input_space=i0)

trainer = SGD(batch_size=128, learning_rate = .01, init_momentum = .5, monitoring_dataset = {'train' : train, 'valid': valid, 'test' : test}, termination_criterion = EpochCounter(max_epochs=20), cost = SumOfCosts([Default(), L1WeightDecay([0.01,0.01,0.01,0.01])]))
                
# watcher = best_params.MonitorBasedSaveBest(
#     channel_name='test_objective',
#     save_path='lpc_phone.pkl')
                  
experiment = Train(dataset=train,
                   model=mdl,
                   algorithm=trainer)

experiment.main_loop()
