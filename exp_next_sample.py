from pylearn2_timit.timitnext import TIMITnext
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
from pylearn2.costs.cost import SumOfCosts
from pylearn2.costs.mlp import *
import cPickle as pickle
import theano
from sys import argv

frame_len = 160
overlap = frame_len-1

#timit_root = argv[1]

train = TIMITnext("train", frame_len, overlap, start=0, stop=100)
valid = TIMITnext("valid", frame_len, overlap, start=0, stop=10)
test = TIMITnext("test", frame_len, overlap, start=0, stop=50)

# Model used to predict the next sample. This model is never trained,
# as it uses parameters predicted by the next model.

i0 = VectorSpace(frame_len)
s0 = Sigmoid(layer_name='h0', dim=500, sparse_init=150)
s1 = Sigmoid(layer_name='h1', dim=250, sparse_init=75)
s2 = Sigmoid(layer_name='h2', dim=100, sparse_init=15)
l0 = Linear(layer_name='y', dim=1, sparse_init=1)

mdl = MLP(layers=[s0, s1, s2, l0], nvis=frame_len, input_space=i0)

trainer = SGD(batch_size=128, learning_rate = .01, init_momentum = .5, monitoring_dataset = {'train' : train, 'valid': valid, 'test' : test}, termination_criterion = EpochCounter(max_epochs=20), cost = SumOfCosts([Default(), L1WeightDecay([0.01,0.01,0.01,0.01])]))
                
watcher = best_params.MonitorBasedSaveBest(
    channel_name='test_objective',
    save_path='nextsample.pkl')
                  
experiment = Train(dataset=train,
                   model=mdl,
                   algorithm=trainer, extensions=[watcher])

experiment.main_loop()





