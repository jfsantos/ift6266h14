from pylearn2_timit.timit import TIMIT
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

num_phones = 61
frame_len = 159

timit_root = argv[1]

train = TIMIT("train", 160, 159)
valid = TIMIT("valid", 160, 159)
test = TIMIT("test", 160, 159)

# Model used to predict the next sample. This model is never trained,
# as it uses parameters predicted by the next model.

i0 = VectorSpace(frame_len+num_phones)
s0 = Sigmoid(layer_name='h0', dim=500, sparse_init=150)
s1 = Sigmoid(layer_name='h1', dim=500, sparse_init=150)
s2 = Sigmoid(layer_name='h2', dim=500, sparse_init=150)
s3 = Sigmoid(layer_name='h3', dim=500, sparse_init=150)
l0 = Linear(layer_name='y', dim=1, sparse_init=150)

mdl = MLP(layers=[s0, s1, s2, s3, l0], nvis=frame_len+num_phones, input_space=i0)

trainer = SGD(batch_size=128, learning_rate = .01, init_momentum = .5, monitoring_dataset = {'train' : train, 'valid': valid, 'test' : test}, termination_criterion = EpochCounter(max_epochs=50), cost = SumOfCosts([Default(), L1WeightDecay([0.01,0.01,0.01,0.01,0.01])]))
                
watcher = best_params.MonitorBasedSaveBest(
    channel_name='test_objective',
    save_path='nextsample_cond_phone.pkl')
                  
experiment = Train(dataset=train,
                   model=mdl,
                   algorithm=trainer, extensions=[watcher])

experiment.main_loop()
