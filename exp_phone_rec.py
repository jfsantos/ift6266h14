from timit_dataset import TimitPhoneData
import itertools
import numpy as np
from pylearn2.datasets import DenseDesignMatrix
from pylearn2.models.mlp import MLP, Sigmoid, Softmax, VectorSpace
from pylearn2.termination_criteria import EpochCounter
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.training_algorithms import learning_rule
from pylearn2.train import Train
from pylearn2.train_extensions import best_params

print "Loading training dataset"
train = TimitPhoneData('/home/jfsantos/data/TIMIT/', framelen=160, overlap=80, start=0, stop=100)
print "Loading validation dataset"
valid = TimitPhoneData('/home/jfsantos/data/TIMIT/', framelen=160, overlap=80, start=2500, stop=2520)
print "Loading test dataset"
test = TimitPhoneData('/home/jfsantos/data/TIMIT/', framelen=160, overlap=80, start=4000, stop=4050)

print "Finished loading datasets"

x0 = VectorSpace(160)
s0 = Sigmoid(layer_name='h0', dim=500, sparse_init=100)
s1 = Sigmoid(layer_name='h1', dim=500, sparse_init=100)
y0 = Softmax(layer_name='y', sparse_init=10, n_classes=61)

mdl = MLP(layers=[s0, s1, y0], nvis=160, input_space=x0)

trainer = SGD(batch_size=1024, learning_rate = .01, init_momentum = .5,
              monitoring_dataset = {'train' : train, 'valid': valid,
                                    'test' : test}, termination_criterion =
              EpochCounter(max_epochs=50))

watcher = best_params.MonitorBasedSaveBest(
    channel_name='valid_y_misclass',
    save_path='phonerec_mlp_2sig_softmax.pkl')
                  
experiment = Train(dataset=train,
                   model=mdl,
                   algorithm=trainer, extensions = [watcher])

experiment.main_loop()


