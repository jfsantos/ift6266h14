import numpy
from pylearn2_timit.timitlpc import TIMITlpc
from pylearn2.space import CompositeSpace, VectorSpace, IndexSpace
from pylearn2.format.target_format import OneHotFormatter

valid = TIMITlpc("valid", frame_length=160, overlap=159, start=10, stop=11)

valid._iter_data_specs = (CompositeSpace((IndexSpace(dim=3,max_labels=61), VectorSpace(dim=10),)), ('phones', 'lpc_features'))

formatter = OneHotFormatter(max_labels=62)

f = lambda x: formatter.format(numpy.asarray(x, dtype=int), mode='merge')

#valid._iter_convert = [f, None]

it = valid.iterator(mode='random_uniform', batch_size=100, num_batches=100)











