# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 19:56:19 2013

@author: Nicholas Léonard
"""

import time, sys

from pylearn2.utils import serial
from itertools import izip
from pylearn2.utils import safe_zip
from collections import OrderedDict
from pylearn2.utils import safe_union

import numpy as np
import theano.sparse as S

from theano.gof.op import get_debug_values
from theano.printing import Print
from theano import function
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import tensor as T
import theano

from pylearn2.linear.matrixmul import MatrixMul

from pylearn2.models.model import Model

from pylearn2.utils import sharedX

from pylearn2.costs.cost import Cost
from pylearn2.costs.mlp import Default
from pylearn2.models.mlp import MLP, Softmax, Layer, Linear
from pylearn2.space import VectorSpace, Conv2DSpace, CompositeSpace, Space
#from pylearn2_objects import MLPCost
 
class Stochastic1Cost(Default):
    def get_gradients(self, model, data, ** kwargs):
        """
        model: a pylearn2 Model instance
        X: a batch in model.get_input_space()
        Y: a batch in model.get_output_space()

        returns: gradients, updates
            gradients:
                a dictionary mapping from the model's parameters
                         to their gradients
                The default implementation is to compute the gradients
                using T.grad applied to the value returned by __call__.
                However, subclasses may return other values for the gradient.
                For example, an intractable cost may return a sampling-based
                approximation to its gradient.
            updates:
                a dictionary mapping shared variables to updates that must
                be applied to them each time these gradients are computed.
                This is to facilitate computation of sampling-based approximate
                gradients.
                The parameters should never appear in the updates dictionary.
                This would imply that computing their gradient changes
                their value, thus making the gradient value outdated.
        """
        try:
            cost = self.expr(model=model, data=data, **kwargs)
        except TypeError,e:
            # If anybody knows how to add type(seslf) to the exception message
            # but still preserve the stack trace, please do so
            # The current code does neither
            e.message += " while calling "+str(type(self))+".__call__"
            print str(type(self))
            print e.message
            raise e

        if cost is None:
            raise NotImplementedError(str(type(self))+" represents an intractable "
                    " cost and does not provide a gradient approximation scheme.")          
        
                   
        layers = model.layers
        
        params = [self.z]
        grads = T.grad(cost, params, disconnected_inputs = 'raise', 
                        consider_constant=model.get_params())

        known_grads = OrderedDict(izip(params, grads))
        rupdates = OrderedDict()
        rgradients = OrderedDict()
        
        '''In reverse order, get gradients one layer at a time.'''
        for layer in reversed(layers):
            gradients, updates \
                = layer.get_gradients(known_grads.copy(), cost)
            known_grads.update(gradients)
            rupdates.update(updates)
            rgradients.update(gradients)
            
        '''print len(rgradients), len(rupdates), len(known_grads)
        
        for param in model.get_params():
            print param.name
            
        print 'grads'
        for (param, grad) in rgradients.iteritems():
            print param.name, grad'''

        return rgradients, rupdates
        
    def get_test_cost(self, model, X, Y):
        state_below = X
        for layer in model.layers:
            if hasattr(layer, 'test_fprop'):
                state_below = layer.test_fprop(state_below)
            else:
                state_below = layer.fprop(state_below)
        y = state_below
        MCE = T.mean(T.cast(T.neq(T.argmax(y, axis=1), 
                       T.argmax(Y, axis=1)), dtype='int32'),
                       dtype=config.floatX)
        return MCE
        

class StochasticBinaryNeuron(Layer):
    """
    Formerly Stochastic2
    
    Stochastic Binary Neuron
    
    A linear layer for the continus part, 
    and two layers with stochastic outputs and non-linear hidden units 
    that generates  a binary mask for the outputs of the continus parts.
    """

    def __init__(self,
                 dim,
                 hidden_dim,
                 layer_name,
                 mean_loss_coeff = 0.5,
                 hidden_activation = 'tanh',
                 sparsity_target = 0.1,
                 sparsity_cost_coeff = 1.0,
                 stoch_grad_coeff = 0.01,
                 linear_activation = None,
                 irange = [None,None,None],
                 istdev = [None,None,None],
                 sparse_init = [None,None,None],
                 sparse_stdev = [1.,1.,1.],
                 init_bias = [0.,0.,0.],
                 W_lr_scale = [None,None,None],
                 b_lr_scale = [None,None,None],
                 max_col_norm = [None,None,None],
                 weight_decay_coeff = [None,None,None]):
        '''
        params
        ------
        dim: 
            number of units on output layer
        hidden_dim: 
            number of units on hidden layer of non-linear part
        mean_loss_coeff: 
            weight of the past moving averages in 
            calculating the moving average vs the weight of the average
            of the current batch.
        hidden_activation:
            activation function used on hidden layer of non-linear part
        sparsity_target:
            target sparsity of the output layer.
        sparsity_cost_coeff:
            coefficient of the sparsity constraint when summing costs
        weight_decay_coeff:
            coefficients of L2 weight decay when summing costs
        other:
            in the lists of params, the first index is for the linear 
            part, while the second and third indices are for the first 
            and second layer of the non-linear part, respectively
        
        '''
                     
        self.__dict__.update(locals())
        del self.self

    def get_lr_scalers(self):
        rval = OrderedDict()

        for i in range(3):
            if self.W_lr_scale[i] is not None:
                rval[self.W[i]] = self.W_lr_scale[i]

            if self.b_lr_scale[i] is not None:
                rval[self.b[i]] = self.b_lr_scale[i]

        return rval

    def set_input_space(self, space):

        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)

        self.output_space = VectorSpace(self.dim)

        self.input_dims = [self.input_dim, self.input_dim, self.hidden_dim]
        self.output_dims = [self.dim, self.hidden_dim, self.dim]
        self.W = [None,None,None]
        self.b = [None,None,None]
        
        for i in range(3):
            self._init_inner_layer(i)
        
        e = 1e-6
        self.mean_loss_deno = sharedX(e+np.zeros((self.output_dims[0],))) 
        
        self.mean_loss_nume = sharedX(e+np.zeros((self.output_dims[0],)))
        
        self.stoch_grad = sharedX(0)
        self.kl_grad = sharedX(0)
        self.linear_grad = sharedX(0)
        
    def _init_inner_layer(self, idx):
        rng = self.mlp.rng
        if self.irange[idx] is not None:
            assert self.istdev[idx] is None
            assert self.sparse_init[idx] is None
            W = rng.uniform(-self.irange[idx], self.irange[idx],
                        (self.input_dims[idx], self.output_dims[idx]))
        elif self.istdev[idx] is not None:
            assert self.sparse_init[idx] is None
            W = rng.randn(self.input_dims[idx], self.output_dims[idx]) \
                    * self.istdev[idx]
        else:
            assert self.sparse_init[idx] is not None
            W = np.zeros((self.input_dims[idx], self.output_dims[idx]))
            for i in xrange(self.output_dims[idx]):
                assert self.sparse_init[idx] <= self.input_dims[idx]
                for j in xrange(self.sparse_init[idx]):
                    idx2 = rng.randint(0, self.input_dims[idx])
                    while W[idx2, i] != 0:
                        idx2 = rng.randint(0, self.input_dims[idx])
                    W[idx2, i] = rng.randn()
            W *= self.sparse_stdev[idx]

        W = sharedX(W)
        W.name = self.layer_name + '_W' + str(idx)
        
        b = sharedX( np.zeros((self.output_dims[idx],)) \
                + self.init_bias[idx], \
                name = self.layer_name + '_b' + str(idx))

        self.W[idx] = W
        self.b[idx] = b


    def censor_updates(self, updates):
        for idx in range(3):
            if self.max_col_norm[idx] is not None:
                W = self.W[idx]
                if W in updates:
                    updated_W = updates[W]
                    col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                    desired_norms = T.clip(col_norms, 0, self.max_col_norm[idx])
                    updates[W] = updated_W * desired_norms / (1e-7 + col_norms)


    def get_params(self):
        rval = [self.W[0], self.W[1], self.W[2], self.b[0], self.b[1], self.b[2]]
        return rval

    def get_weights(self):
        rval = []
        for i in range(3):
            W = self.W[i]
            rval.append(W.get_value())
            
        return rval

    def set_weights(self, weights):
        for i in range(3):
            W = self.W[i]
            W.set_value(weights[i])

    def set_biases(self, biases):
        for i in range(3):
            self.b[i].set_value(biases[i])

    def get_biases(self):
        rval = []
        for i in range(3):
            rval.append(self.b[i].get_value())
        return rval

    def get_weights_format(self):
        return ('v', 'h')

    def get_weights_topo(self):
        raise NotImplementedError()
        
    def get_monitoring_channels(self):
        rval = OrderedDict([
                ('mean_loss_nume_mean',  self.mean_loss_nume.mean()),
                ('mean_loss_deno_mean',  self.mean_loss_deno.mean()),
            ])
        rval['stoch_grad'] = self.stoch_grad
        rval['kl_grad'] = self.kl_grad
        rval['linear_grad'] = self.linear_grad
        
        for i in range(3):
            sq_W = T.sqr(self.W[i])

            row_norms = T.sqrt(sq_W.sum(axis=1))
            col_norms = T.sqrt(sq_W.sum(axis=0))
            
            rval['row_norms_max'+str(i)] = row_norms.max()
            rval['col_norms_max'+str(i)] = col_norms.max()
        
        return rval
        
    def get_monitoring_channels_from_state(self, state, target=None):
        rval =  OrderedDict()
        # sparisty of outputs:
        rval['mean_output_sparsity'] = self.m_mean.mean()
        # proportion of sigmoids that have prob > 0.5
        # good when equal to sparsity
        floatX = theano.config.floatX
        rval['mean_sparsity_prop'] \
            = T.cast(T.gt(self.m_mean, 0.5),floatX).mean()
        # same as above but for intermediate thresholds:
        rval['mean_sparsity_prop0.2'] \
            = T.cast(T.gt(self.m_mean, 0.2),floatX).mean()
        rval['mean_sparsity_prop0.3'] \
            = T.cast(T.gt(self.m_mean, 0.3),floatX).mean()
        rval['mean_sparsity_prop0.4'] \
            = T.cast(T.gt(self.m_mean, 0.4),floatX).mean()    
        # or just plain standard deviation (less is bad):
        rval['output_stdev'] = self.m_mean.std()
        # stdev of unit stdevs (more is bad)
        rval['output_meta_stdev'] = self.m_mean.std(axis=0).std()
        # max and min proportion of these probs per unit
        prop_per_unit = T.cast(T.gt(self.m_mean, 0.5),floatX).mean(0)
        # if this is high, it means a unit is likely always active (bad)
        rval['max_unit_sparsity_prop'] = prop_per_unit.max()
        rval['min_unit_sparsity_prop'] = prop_per_unit.min()
        # in both cases, high means units are popular (bad)
        # proportion of units with p>0.5 more than 50% of time:
        rval['mean_unit_sparsity_meta_prop'] \
            = T.cast(T.gt(prop_per_unit,0.5),floatX).mean()
        # proportion of units with p>0.5 more than 75% of time:
        rval['mean_unit_sparsity_meta_prop2'] \
            = T.cast(T.gt(prop_per_unit,0.75),floatX).mean()
        
        return rval

    def fprop(self, state_below, threshold=None, stochastic=True):
        self.input_space.validate(state_below)

        if self.requires_reformat:
            if not isinstance(state_below, tuple):
                for sb in get_debug_values(state_below):
                    if sb.shape[0] != self.dbm.batch_size:
                        raise ValueError("self.dbm.batch_size is %d but got shape of %d" % (self.dbm.batch_size, sb.shape[0]))
                    assert reduce(lambda x,y: x * y, sb.shape[1:]) == self.input_dim

            state_below = self.input_space.format_as(state_below, self.desired_space)
        
        self.x = state_below
        
        # linear part
        if isinstance(self.x, S.SparseVariable):
            z = S.dot(self.x,self.W[0]) + self.b[0]
        else:
            z = T.dot(self.x,self.W[0]) + self.b[0]
        
        self.output_activation = None
        # activate hidden units of non-linear part
        if self.output_activation is None:
            self.z = z
        elif self.hidden_activation == 'tanh':
            self.z = T.tanh(z)
        elif self.output_activation == 'sigmoid':
            self.z = T.nnet.sigmoid(z)
        elif self.output_activation == 'softmax':
            self.z = T.nnet.softmax(z)
        elif self.output_activation == 'rectifiedlinear':
            self.z = T.maximum(0, z)
        else:
            raise NotImplementedError()
        
        # first layer non-linear part
        if isinstance(self.x, S.SparseVariable):
            h = S.dot(self.x,self.W[1]) + self.b[1]
        else:
            h = T.dot(self.x,self.W[1]) + self.b[1]
        
        # activate hidden units of non-linear part
        if self.hidden_activation is None:
            self.h = h
        elif self.hidden_activation == 'tanh':
            self.h = T.tanh(h)
        elif self.hidden_activation == 'sigmoid':
            self.h = T.nnet.sigmoid(h)
        elif self.hidden_activation == 'softmax':
            self.h = T.nnet.softmax(h)
        elif self.hidden_activation == 'rectifiedlinear':
            self.h = T.maximum(0, h)
        else:
            raise NotImplementedError()
        
        # second layer non-linear part
        self.a = T.dot(self.h,self.W[2]) + self.b[2]
        
        # activate non-linear part to get bernouilli probabilities
        self.m_mean = T.nnet.sigmoid(self.a)
        
        if threshold is None:
            if stochastic:
                # sample from bernouili probs to generate a mask
                rng = MRG_RandomStreams(self.mlp.rng.randint(2**15))
                self.m = rng.binomial(size = self.m_mean.shape, n = 1, 
                        p = self.m_mean, dtype=self.m_mean.type.dtype)
            else:
                self.m = self.m_mean
        else:
            # deterministic mask:
            self.m = T.cast(T.gt(self.m_mean, threshold), \
                                        theano.config.floatX)
           
        # mask output of linear part with samples from linear part
        self.p = self.m * self.z
        
        if self.layer_name is not None:
            self.z.name = self.layer_name + '_z'
            self.h.name = self.layer_name + '_h'
            self.a.name = self.layer_name + '_a'
            self.m_mean.name = self.layer_name + '_m_mean'
            self.m.name = self.layer_name + '_m'
            self.p.name = self.layer_name + '_p'
        
        return self.p
        
    def test_fprop(self, state_below, threshold=None, stochastic=True):
        return self.fprop(state_below, threshold, stochastic)
        
    def cost(self, Y, Y_hat):
        return self.cost_from_cost_matrix(self.cost_matrix(Y, Y_hat))

    def cost_from_cost_matrix(self, cost_matrix):
        return cost_matrix.sum(axis=1).mean()

    def cost_matrix(self, Y, Y_hat):
        return T.sqr(Y - Y_hat)
        
    def get_gradients(self, known_grads, loss):
        '''
        Computes gradients and updates for this layer given the known
        gradients of the upper layers, and the vector of losses for the
        batch.
        '''
        updates = OrderedDict()
        
        cost = self.get_kl_divergence() + self.get_weight_decay()
        # gradient of linear part.
        params = [self.W[0], self.b[0]]
        grads = T.grad(cost=None, wrt=params, known_grads=known_grads, 
                        consider_constant=[self.m, self.x],
                        disconnected_inputs='raise')
        cost_grads = T.grad(cost=cost, wrt=params,
                        consider_constant=[self.m, self.x],
                        disconnected_inputs='ignore')
                    
        updates[self.linear_grad] = T.abs_(grads[0]).mean()
        
        for i in range(len(grads)):
            grads[i] += cost_grads[i]
            
        gradients = OrderedDict(izip(params, grads))
        
        # update moving average loss for each unit where 1 was sampled
        loss = loss.dimshuffle(0,'x')
        
        delta = (self.m - self.m_mean)
        
        updates[self.mean_loss_nume] = \
                (self.mean_loss_coeff * self.mean_loss_nume) \
                + ((1. - self.mean_loss_coeff) * \
                    (T.sqr(delta) * loss).mean(axis=0))
                    
        updates[self.mean_loss_deno] = \
                (self.mean_loss_coeff * self.mean_loss_deno) \
                + ((1. - self.mean_loss_coeff) * \
                    T.sqr(delta).mean(axis=0))
        
        
        # gradients of non-linear part.
        ## obtain a lower-variance unbiased estimator by using 
        ## separate moving averages of the loss for each unit
        mean_loss = self.mean_loss_nume/self.mean_loss_deno
        known_grads[self.a] = \
            self.stoch_grad_coeff \
            * delta * (loss - mean_loss.dimshuffle('x',0))
            
        params = [self.W[1],self.W[2],self.b[1],self.b[2]]
        
        grads = T.grad(cost=None, wrt=params, known_grads=known_grads,
                       consider_constant=[self.z, self.x],
                       disconnected_inputs='raise')
                       
        updates[self.stoch_grad] = T.abs_(grads[1]).mean()
    
        cost_grads = T.grad(cost=cost, wrt=params,
                       consider_constant=[self.z, self.x],
                       disconnected_inputs='ignore')
                       
        updates[self.kl_grad] = T.abs_(cost_grads[1]).mean()
                       
        for i in range(len(grads)):
            grads[i] += cost_grads[i]
                       
        gradients.update(OrderedDict(izip(params, grads)))
        
        return gradients, updates
        
    def get_kl_divergence(self):
        '''
        Minimize KL-divergence of unit binomial distributions with 
        binomial distribution of probability self.sparsity_target.
        This could also be modified to keep a running average of unit 
        samples
        '''
        e = 1e-6
        cost = - self.sparsity_cost_coeff * ( \
                (self.sparsity_target * T.log(e+self.m_mean.mean(axis=0))) \
                +((1.-self.sparsity_target) * T.log(e+(1.-self.m_mean.mean(axis=0)))) \
             ).sum()
        return cost
        
    def get_weight_decay(self):
        rval = 0
        for i in range(3):
            if self.weight_decay_coeff[i] is not None:
                rval += self.weight_decay_coeff[i]*T.sqr(self.W[i]).sum()
        return rval

class GaterOnly(Layer):
    """
    No experts.
    Formerly Stochastic3
    One tanh layer followed by a stochastic sigmoid layer, they both 
    learn using unbiased estimator of the gradient.
    """

    def __init__(self,
                 dim,
                 hidden_dim,
                 layer_name,
                 mean_loss_coeff = 0.9,
                 hidden_activation = 'tanh',
                 sparsity_target = 0.1,
                 sparsity_cost_coeff = 1.0,
                 stoch_grad_coeff = 0.01,
                 linear_activation = None,
                 irange = [None,None],
                 istdev = [None,None],
                 sparse_init = [None,None],
                 sparse_stdev = [1.,1.],
                 init_bias = [0.,0.],
                 W_lr_scale = [None,None],
                 b_lr_scale = [None,None],
                 max_col_norm = [None,None],
                 weight_decay_coeff = [None,None]):
        '''
        params
        ------
        dim: 
            number of units on output layer
        hidden_dim: 
            number of units on hidden layer
        mean_loss_coeff: 
            weight of the past moving averages in 
            calculating the moving average vs the weight of the average
            of the current batch.
        hidden_activation:
            activation function used on hidden layer of non-linear part
        sparsity_target:
            target sparsity of the output layer.
        sparsity_cost_coeff:
            coefficient of the sparsity constraint when summing costs
        weight_decay_coeff:
            coefficients of L2 weight decay when summing costs
        other:
            in the lists of params, the first index is for the linear 
            part, while the second and third indices are for the first 
            and second layer of the non-linear part, respectively
        
        '''
                     
        self.__dict__.update(locals())
        del self.self

    def get_lr_scalers(self):
        rval = OrderedDict()

        for i in range(2):
            if self.W_lr_scale[i] is not None:
                rval[self.W[i]] = self.W_lr_scale[i]

            if self.b_lr_scale[i] is not None:
                rval[self.b[i]] = self.b_lr_scale[i]

        return rval

    def set_input_space(self, space):

        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)

        self.output_space = VectorSpace(self.dim)

        self.input_dims = [self.input_dim, self.hidden_dim]
        self.output_dims = [self.hidden_dim, self.dim]
        self.W = [None,None]
        self.b = [None,None]
        
        for i in range(2):
            self._init_inner_layer(i)
        
        e = 1e-6
        self.mean_loss_deno = sharedX(e+np.zeros((self.output_dims[1],))) 
        
        self.mean_loss_nume = sharedX(e+np.zeros((self.output_dims[1],)))
        
        self.stoch_grad = sharedX(0)
        self.kl_grad = sharedX(0)
        
    def _init_inner_layer(self, idx):
        rng = self.mlp.rng
        if self.irange[idx] is not None:
            assert self.istdev[idx] is None
            assert self.sparse_init[idx] is None
            W = rng.uniform(-self.irange[idx], self.irange[idx],
                        (self.input_dims[idx], self.output_dims[idx]))
        elif self.istdev[idx] is not None:
            assert self.sparse_init[idx] is None
            W = rng.randn(self.input_dims[idx], self.output_dims[idx]) \
                    * self.istdev[idx]
        else:
            assert self.sparse_init[idx] is not None
            W = np.zeros((self.input_dims[idx], self.output_dims[idx]))
            for i in xrange(self.output_dims[idx]):
                assert self.sparse_init[idx] <= self.input_dims[idx]
                for j in xrange(self.sparse_init[idx]):
                    idx2 = rng.randint(0, self.input_dims[idx])
                    while W[idx2, i] != 0:
                        idx2 = rng.randint(0, self.input_dims[idx])
                    W[idx2, i] = rng.randn()
            W *= self.sparse_stdev[idx]

        W = sharedX(W)
        W.name = self.layer_name + '_W' + str(idx)
        
        b = sharedX( np.zeros((self.output_dims[idx],)) \
                + self.init_bias[idx], \
                name = self.layer_name + '_b' + str(idx))

        self.W[idx] = W
        self.b[idx] = b


    def censor_updates(self, updates):
        for idx in range(2):
            if self.max_col_norm[idx] is not None:
                W = self.W[idx]
                if W in updates:
                    updated_W = updates[W]
                    col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                    desired_norms = T.clip(col_norms, 0, self.max_col_norm[idx])
                    updates[W] = updated_W * desired_norms / (1e-7 + col_norms)


    def get_params(self):
        rval = [self.W[0], self.W[1], self.b[0], self.b[1]]
        return rval

    def get_weights(self):
        rval = []
        for i in range(2):
            W = self.W[i]
            rval.append(W.get_value())
            
        return rval

    def set_weights(self, weights):
        for i in range(2):
            W = self.W[i]
            W.set_value(weights[i])

    def set_biases(self, biases):
        for i in range(2):
            self.b[i].set_value(biases[i])

    def get_biases(self):
        rval = []
        for i in range(2):
            rval.append(self.b[i].get_value())
        return rval

    def get_weights_format(self):
        return ('v', 'h')

    def get_weights_topo(self):
        raise NotImplementedError()
        
    def get_monitoring_channels(self):
        rval = OrderedDict([
                ('mean_loss_nume_mean',  self.mean_loss_nume.mean()),
                ('mean_loss_deno_mean',  self.mean_loss_deno.mean()),
            ])
        rval['stoch_grad'] = self.stoch_grad
        rval['kl_grad'] = self.kl_grad
        
        for i in range(2):
            sq_W = T.sqr(self.W[i])

            row_norms = T.sqrt(sq_W.sum(axis=1))
            col_norms = T.sqrt(sq_W.sum(axis=0))
            
            rval['row_norms_max'+str(i)] = row_norms.max()
            rval['col_norms_max'+str(i)] = col_norms.max()
        
        return rval
        
    def get_monitoring_channels_from_state(self, state, target=None):
        rval =  OrderedDict()
        # sparisty of outputs:
        rval['mean_output_sparsity'] = self.m_mean.mean()
        # proportion of sigmoids that have prob > 0.5
        # good when equal to sparsity
        floatX = theano.config.floatX
        rval['mean_sparsity_prop'] \
            = T.cast(T.gt(self.m_mean, 0.5),floatX).mean()
        # same as above but for intermediate thresholds:
        rval['mean_sparsity_prop0.2'] \
            = T.cast(T.gt(self.m_mean, 0.2),floatX).mean()
        rval['mean_sparsity_prop0.3'] \
            = T.cast(T.gt(self.m_mean, 0.3),floatX).mean()
        rval['mean_sparsity_prop0.4'] \
            = T.cast(T.gt(self.m_mean, 0.4),floatX).mean()    
        # or just plain standard deviation (less is bad):
        rval['output_stdev'] = self.m_mean.std()
        # stdev of unit stdevs (more is bad)
        rval['output_meta_stdev'] = self.m_mean.std(axis=0).std()
        # max and min proportion of these probs per unit
        prop_per_unit = T.cast(T.gt(self.m_mean, 0.5),floatX).mean(0)
        # if this is high, it means a unit is likely always active (bad)
        rval['max_unit_sparsity_prop'] = prop_per_unit.max()
        rval['min_unit_sparsity_prop'] = prop_per_unit.min()
        # in both cases, high means units are popular (bad)
        # proportion of units with p>0.5 more than 50% of time:
        rval['mean_unit_sparsity_meta_prop'] \
            = T.cast(T.gt(prop_per_unit,0.5),floatX).mean()
        # proportion of units with p>0.5 more than 75% of time:
        rval['mean_unit_sparsity_meta_prop2'] \
            = T.cast(T.gt(prop_per_unit,0.75),floatX).mean()
        
        return rval

    def fprop(self, state_below, threshold=None, stochastic=True):
        self.input_space.validate(state_below)

        if self.requires_reformat:
            if not isinstance(state_below, tuple):
                for sb in get_debug_values(state_below):
                    if sb.shape[0] != self.dbm.batch_size:
                        raise ValueError("self.dbm.batch_size is %d but got shape of %d" % (self.dbm.batch_size, sb.shape[0]))
                    assert reduce(lambda x,y: x * y, sb.shape[1:]) == self.input_dim

            state_below = self.input_space.format_as(state_below, self.desired_space)
        
        self.x = state_below
        
        # first layer
        if isinstance(self.x, S.SparseVariable):
            h = S.dot(self.x,self.W[0]) + self.b[0]
        else:
            h = T.dot(self.x,self.W[0]) + self.b[0]
        
        # activate hidden units
        if self.hidden_activation is None:
            self.h = h
        elif self.hidden_activation == 'tanh':
            self.h = T.tanh(h)
        elif self.hidden_activation == 'sigmoid':
            self.h = T.nnet.sigmoid(h)
        elif self.hidden_activation == 'softmax':
            self.h = T.nnet.softmax(h)
        elif self.hidden_activation == 'rectifiedlinear':
            self.h = T.maximum(0, h)
        else:
            raise NotImplementedError()
        
        # second layer
        self.a = T.dot(self.h,self.W[1]) + self.b[1]
        
        # activate non-linear part to get bernouilli probabilities
        self.m_mean = T.nnet.sigmoid(self.a)
        
        if threshold is None:
            if stochastic:
                # sample from bernouili probs to generate a mask
                rng = MRG_RandomStreams(self.mlp.rng.randint(2**15))
                self.m = rng.binomial(size = self.m_mean.shape, n = 1, 
                        p = self.m_mean, dtype=self.m_mean.type.dtype)
                '''uniform = rng.uniform(size = self.m_mean.shape, dtype=self.m_mean.type.dtype)
                self.m = T.cast(T.gt(uniform,self.m_mean),dtype=theano.config.floatX)'''
            else:
                self.m = self.m_mean
        else:
            # deterministic mask:
            self.m = T.cast(T.gt(self.m_mean, threshold), \
                                        theano.config.floatX)
        
        if self.layer_name is not None:
            self.h.name = self.layer_name + '_h'
            self.a.name = self.layer_name + '_a'
            self.m_mean.name = self.layer_name + '_m_mean'
            self.m.name = self.layer_name + '_m'
        
        return self.m
        
    def test_fprop(self, state_below, threshold=None, stochastic=True):
        return self.fprop(state_below, threshold, stochastic)

    def cost(self, Y, Y_hat):
        return self.cost_from_cost_matrix(self.cost_matrix(Y, Y_hat))

    def cost_from_cost_matrix(self, cost_matrix):
        return cost_matrix.sum(axis=1).mean()

    def cost_matrix(self, Y, Y_hat):
        return T.sqr(Y - Y_hat)
        
    def get_gradients(self, known_grads, loss):
        '''
        Computes gradients and updates for this layer given the vector 
        of losses for the batch.
        '''
        updates = OrderedDict()            
        gradients = OrderedDict()
        
        cost = self.get_kl_divergence() + self.get_weight_decay()
        
        # update moving average loss for each unit where 1 was sampled
        loss = loss.dimshuffle(0,'x')
        
        delta = (self.m - self.m_mean)
        
        updates[self.mean_loss_nume] = \
                (self.mean_loss_coeff * self.mean_loss_nume) \
                + ((1. - self.mean_loss_coeff) * \
                    (T.sqr(delta) * loss).mean(axis=0))
                    
        updates[self.mean_loss_deno] = \
                (self.mean_loss_coeff * self.mean_loss_deno) \
                + ((1. - self.mean_loss_coeff) * \
                    T.sqr(delta).mean(axis=0))
        
        
        # gradients of non-linear part.
        ## obtain a lower-variance unbiased estimator by using 
        ## separate moving averages of the loss for each unit
        mean_loss = self.mean_loss_nume/self.mean_loss_deno
        known_grads[self.a] = \
            self.stoch_grad_coeff \
            * delta * (loss - mean_loss.dimshuffle('x',0))
            
        params = [self.W[0],self.W[1],self.b[0],self.b[1]]
        
        grads = T.grad(cost=None, wrt=params, known_grads=known_grads,
                       consider_constant=[self.x],
                       disconnected_inputs='raise')
                       
        updates[self.stoch_grad] = T.abs_(grads[0]).mean()
    
        cost_grads = T.grad(cost=cost, wrt=params,
                       consider_constant=[self.x],
                       disconnected_inputs='ignore')
                       
        updates[self.kl_grad] = T.abs_(cost_grads[0]).mean()
                       
        for i in range(len(grads)):
            grads[i] += cost_grads[i]
                       
        gradients.update(OrderedDict(izip(params, grads)))
        
        return gradients, updates
        
    def get_kl_divergence(self):
        '''
        Minimize KL-divergence of unit binomial distributions with 
        binomial distribution of probability self.sparsity_target.
        This could also be modified to keep a running average of unit 
        samples
        '''
        e = 1e-6
        cost = - self.sparsity_cost_coeff * ( \
                (self.sparsity_target * T.log(e+self.m_mean.mean(axis=0))) \
                +((1.-self.sparsity_target) * T.log(e+(1.-self.m_mean.mean(axis=0)))) \
             ).sum()
        return cost
        
    def get_weight_decay(self):
        rval = 0
        for i in range(2):
            if self.weight_decay_coeff[i] is not None:
                rval += self.weight_decay_coeff[i]*T.sqr(self.W[i]).sum()
        return rval
  
                       
class StochasticSoftmax(Softmax):
    def __init__(self, n_classes, layer_name, irange = None,
                 istdev = None,
                 sparse_init = None, W_lr_scale = None,
                 b_lr_scale = None, max_row_norm = None,
                 no_affine = False,
                 max_col_norm = None, init_bias_target_marginals= None,
                 weight_decay_coeff = None):
        """
        """
        self.weight_decay_coeff = weight_decay_coeff
        Softmax.__init__(self, n_classes, layer_name,irange,istdev,
                 sparse_init,W_lr_scale,b_lr_scale,max_row_norm,
                 no_affine,max_col_norm,init_bias_target_marginals)
                 
    def get_weight_decay(self):
        if self.weight_decay_coeff is None:
            return None
        return self.weight_decay_coeff * T.sqr(self.W).sum()
        
    def fprop(self, state_below):

        self.input_space.validate(state_below)

        if self.needs_reformat:
            state_below = self.input_space.format_as(state_below, self.desired_space)

        for value in get_debug_values(state_below):
            if self.mlp.batch_size is not None and value.shape[0] != self.mlp.batch_size:
                raise ValueError("state_below should have batch size "+str(self.dbm.batch_size)+" but has "+str(value.shape[0]))

        self.desired_space.validate(state_below)
        assert state_below.ndim == 2
        
        self.x = state_below

        if not hasattr(self, 'no_affine'):
            self.no_affine = False

        if self.no_affine:
            Z = state_below
        else:
            assert self.W.ndim == 2
            b = self.b

            Z = T.dot(state_below, self.W) + b

        rval = T.nnet.softmax(Z)

        for value in get_debug_values(rval):
            if self.mlp.batch_size is not None:
                assert value.shape[0] == self.mlp.batch_size

        return rval
        
    def get_gradients(self, known_grads, loss):
        params = self.get_params()
        cost = self.get_weight_decay()
        
        grads = T.grad(cost=None, wrt=params, known_grads=known_grads, 
                        consider_constant=[self.x],
                        disconnected_inputs='raise')
        if cost is not None:
            cost_grads = T.grad(cost=cost, wrt=params,
                            consider_constant=[self.x],
                            disconnected_inputs='raise')
                           
            for i in range(len(grads)):
                grads[i] += cost_grads[i]
                        
        gradients = OrderedDict(izip(params, grads))
        
        return gradients, OrderedDict()
        
        
class SparseTanh(Linear):
    """
    Implementation of the tanh nonlinearity for MLP.
    """

    def _linear_part(self, state_below):
        # TODO: Refactor More Better(tm)
        self.input_space.validate(state_below)

        if self.requires_reformat:
            if not isinstance(state_below, tuple):
                for sb in get_debug_values(state_below):
                    if sb.shape[0] != self.dbm.batch_size:
                        raise ValueError("self.dbm.batch_size is %d but got shape of %d" % (self.dbm.batch_size, sb.shape[0]))
                    assert reduce(lambda x,y: x * y, sb.shape[1:]) == self.input_dim

            state_below = self.input_space.format_as(state_below, self.desired_space)

        if self.softmax_columns:
            W, = self.transformer.get_params()
            W = W.T
            W = T.nnet.softmax(W)
            W = W.T
            z = S.dot(state_below, W) + self.b
        else:
            W, = self.transformer.get_params()
            z = S.dot(state_below, W) + self.b
        if self.layer_name is not None:
            z.name = self.layer_name + '_z'
        if self.copy_input:
            z = T.concatenate((z, state_below), axis=1)
        return z
        
    def fprop(self, state_below):
        p = self._linear_part(state_below)
        p = T.tanh(p)
        return p

    def cost(self, *args, **kwargs):
        raise NotImplementedError()
        
   
class StraightThrough(Layer):
    """
    Formerly Stochastic4
    
    Biased low-variance estimator
    
    Straight-Through
    """

    def __init__(self,
                 dim,
                 hidden_dim,
                 layer_name,
                 hidden_activation = 'tanh',
                 expert_activation = 'linear',
                 derive_sigmoid = True,
                 sparsity_target = 0.1,
                 sparsity_cost_coeff = 1.0,
                 irange = [None,None,None],
                 istdev = [None,None,None],
                 sparse_init = [None,None,None],
                 sparse_stdev = [1.,1.,1.],
                 init_bias = [0.,0.,0.],
                 W_lr_scale = [None,None,None],
                 b_lr_scale = [None,None,None],
                 max_col_norm = [None,None,None],
                 weight_decay_coeff = [None,None,None]):
        '''
        params
        ------
        dim: 
            number of units on output layer
        hidden_dim: 
            number of units on hidden layer of non-linear part
        hidden_activation:
            activation function used on hidden layer of non-linear part
        sparsity_target:
            target sparsity of the output layer.
        sparsity_cost_coeff:
            coefficient of the sparsity constraint when summing costs
        weight_decay_coeff:
            coefficients of L2 weight decay when summing costs
        other:
            in the lists of params, the first index is for the linear 
            part, while the second and third indices are for the first 
            and second layer of the non-linear part, respectively
        
        '''
                     
        self.__dict__.update(locals())
        del self.self

    def get_lr_scalers(self):
        rval = OrderedDict()

        for i in range(3):
            if self.W_lr_scale[i] is not None:
                rval[self.W[i]] = self.W_lr_scale[i]

            if self.b_lr_scale[i] is not None:
                rval[self.b[i]] = self.b_lr_scale[i]

        return rval
        
    def activate(self, x, function_name):
        if (function_name is None) or (function_name == 'linear'):
            y = x
        elif function_name == 'tanh':
            y = T.tanh(x)
        elif function_name == 'sigmoid':
            y = T.nnet.sigmoid(x)
        elif function_name == 'softmax':
            y = T.nnet.softmax(x)
        elif function_name == 'rectifiedlinear':
            y = T.maximum(0, x)
        elif function_name == 'softplus':
            y = T.nnet.softplus(x)
        elif function_name == 'softmax':
            y = T.nnet.softmax(x)
        elif function_name == 'rectifiedsoftplus':
            y = T.nnet.softplus(T.maximum(0, x))
        else:
            raise NotImplementedError()
        return y

    def set_input_space(self, space):

        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)

        self.output_space = VectorSpace(self.dim)

        self.input_dims = [self.input_dim, self.input_dim, self.hidden_dim]
        self.output_dims = [self.dim, self.hidden_dim, self.dim]
        self.W = [None,None,None]
        self.b = [None,None,None]
        
        for i in range(3):
            self._init_inner_layer(i)
        
        self.stoch_grad = sharedX(0)
        self.kl_grad = sharedX(0)
        self.linear_grad = sharedX(0)
        
    def _init_inner_layer(self, idx):
        rng = self.mlp.rng
        if self.irange[idx] is not None:
            assert self.istdev[idx] is None
            assert self.sparse_init[idx] is None
            W = rng.uniform(-self.irange[idx], self.irange[idx],
                        (self.input_dims[idx], self.output_dims[idx]))
        elif self.istdev[idx] is not None:
            assert self.sparse_init[idx] is None
            W = rng.randn(self.input_dims[idx], self.output_dims[idx]) \
                    * self.istdev[idx]
        else:
            assert self.sparse_init[idx] is not None
            W = np.zeros((self.input_dims[idx], self.output_dims[idx]))
            for i in xrange(self.output_dims[idx]):
                assert self.sparse_init[idx] <= self.input_dims[idx]
                for j in xrange(self.sparse_init[idx]):
                    idx2 = rng.randint(0, self.input_dims[idx])
                    while W[idx2, i] != 0:
                        idx2 = rng.randint(0, self.input_dims[idx])
                    W[idx2, i] = rng.randn()
            W *= self.sparse_stdev[idx]

        W = sharedX(W)
        W.name = self.layer_name + '_W' + str(idx)
        
        b = sharedX( np.zeros((self.output_dims[idx],)) \
                + self.init_bias[idx], \
                name = self.layer_name + '_b' + str(idx))

        self.W[idx] = W
        self.b[idx] = b


    def censor_updates(self, updates):
        for idx in range(3):
            if self.max_col_norm[idx] is not None:
                W = self.W[idx]
                if W in updates:
                    updated_W = updates[W]
                    col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                    desired_norms = T.clip(col_norms, 0, self.max_col_norm[idx])
                    updates[W] = updated_W * desired_norms / (1e-7 + col_norms)


    def get_params(self):
        rval = [self.W[0], self.W[1], self.W[2], self.b[0], self.b[1], self.b[2]]
        return rval

    def get_weights(self):
        rval = []
        for i in range(3):
            W = self.W[i]
            rval.append(W.get_value())
            
        return rval

    def set_weights(self, weights):
        for i in range(3):
            W = self.W[i]
            W.set_value(weights[i])

    def set_biases(self, biases):
        for i in range(3):
            self.b[i].set_value(biases[i])

    def get_biases(self):
        rval = []
        for i in range(3):
            rval.append(self.b[i].get_value())
        return rval

    def get_weights_format(self):
        return ('v', 'h')

    def get_weights_topo(self):
        raise NotImplementedError()
        
    def get_monitoring_channels(self):
        rval = OrderedDict()
        rval['stoch_grad'] = self.stoch_grad
        rval['kl_grad'] = self.kl_grad
        rval['linear_grad'] = self.linear_grad
        
        for i in range(3):
            sq_W = T.sqr(self.W[i])

            row_norms = T.sqrt(sq_W.sum(axis=1))
            col_norms = T.sqrt(sq_W.sum(axis=0))
            
            rval['row_norms_max'+str(i)] = row_norms.max()
            rval['col_norms_max'+str(i)] = col_norms.max()
        
        return rval
        
    def get_monitoring_channels_from_state(self, state, target=None):
        rval =  OrderedDict()
        # sparisty of outputs:
        rval['mean_output_sparsity'] = self.m_mean.mean()
        # proportion of sigmoids that have prob > 0.5
        # good when equal to sparsity
        floatX = theano.config.floatX
        rval['mean_sparsity_prop'] \
            = T.cast(T.gt(self.m_mean, 0.5),floatX).mean()
        # same as above but for intermediate thresholds:
        rval['mean_sparsity_prop0.2'] \
            = T.cast(T.gt(self.m_mean, 0.2),floatX).mean()
        rval['mean_sparsity_prop0.3'] \
            = T.cast(T.gt(self.m_mean, 0.3),floatX).mean()
        rval['mean_sparsity_prop0.4'] \
            = T.cast(T.gt(self.m_mean, 0.4),floatX).mean()    
        # or just plain standard deviation (less is bad):
        rval['output_stdev'] = self.m_mean.std()
        # stdev of unit stdevs (more is bad)
        rval['output_meta_stdev'] = self.m_mean.std(axis=0).std()
        # max and min proportion of these probs per unit
        prop_per_unit = T.cast(T.gt(self.m_mean, 0.5),floatX).mean(0)
        # if this is high, it means a unit is likely always active (bad)
        rval['max_unit_sparsity_prop'] = prop_per_unit.max()
        rval['min_unit_sparsity_prop'] = prop_per_unit.min()
        # in both cases, high means units are popular (bad)
        # proportion of units with p>0.5 more than 50% of time:
        rval['mean_unit_sparsity_meta_prop'] \
            = T.cast(T.gt(prop_per_unit,0.5),floatX).mean()
        # proportion of units with p>0.5 more than 75% of time:
        rval['mean_unit_sparsity_meta_prop2'] \
            = T.cast(T.gt(prop_per_unit,0.75),floatX).mean()
        
        return rval

    def fprop(self, state_below, threshold=None, stochastic=True):
        self.input_space.validate(state_below)

        if self.requires_reformat:
            if not isinstance(state_below, tuple):
                for sb in get_debug_values(state_below):
                    if sb.shape[0] != self.dbm.batch_size:
                        raise ValueError("self.dbm.batch_size is %d but got shape of %d" % (self.dbm.batch_size, sb.shape[0]))
                    assert reduce(lambda x,y: x * y, sb.shape[1:]) == self.input_dim

            state_below = self.input_space.format_as(state_below, self.desired_space)
        
        self.x = state_below
        
        # linear part
        if isinstance(self.x, S.SparseVariable):
            z = S.dot(self.x,self.W[0]) + self.b[0]
        else:
            z = T.dot(self.x,self.W[0]) + self.b[0]
        
        self.z = self.activate(z, self.expert_activation)
        
        # first layer non-linear part
        if isinstance(self.x, S.SparseVariable):
            h = S.dot(self.x,self.W[1]) + self.b[1]
        else:
            h = T.dot(self.x,self.W[1]) + self.b[1]
        
        self.h = self.activate(h, self.hidden_activation)
        
        
        # second layer non-linear part
        self.a = T.dot(self.h,self.W[2]) + self.b[2]
        
        # activate non-linear part to get bernouilli probabilities
        self.m_mean = T.nnet.sigmoid(self.a)
        
        if threshold is None:
            if stochastic:
                # sample from bernouili probs to generate a mask
                rng = MRG_RandomStreams(self.mlp.rng.randint(2**15))
                self.m = rng.binomial(size = self.m_mean.shape, n = 1, 
                        p = self.m_mean, dtype=self.m_mean.type.dtype)
            else:
                self.m = self.m_mean
        else:
            # deterministic mask:
            self.m = T.cast(T.gt(self.m_mean, threshold), \
                                        theano.config.floatX)
           
        # mask output of linear part with samples from linear part
        self.p = self.m * self.z
        
        if self.layer_name is not None:
            self.z.name = self.layer_name + '_z'
            self.h.name = self.layer_name + '_h'
            self.a.name = self.layer_name + '_a'
            self.m_mean.name = self.layer_name + '_m_mean'
            self.m.name = self.layer_name + '_m'
            self.p.name = self.layer_name + '_p'
        
        return self.p
        
    def test_fprop(self, state_below, threshold=None, stochastic=True):
        return self.fprop(state_below, threshold, stochastic)
        
    def cost(self, Y, Y_hat):
        return self.cost_from_cost_matrix(self.cost_matrix(Y, Y_hat))

    def cost_from_cost_matrix(self, cost_matrix):
        return cost_matrix.sum(axis=1).mean()

    def cost_matrix(self, Y, Y_hat):
        return T.sqr(Y - Y_hat)
        
    def get_gradients(self, known_grads, loss):
        '''
        Computes gradients and updates for this layer given the known
        gradients of the upper layers, and the vector of losses for the
        batch.
        '''
        updates = OrderedDict()
        
        cost = self.get_kl_divergence() + self.get_weight_decay()
        # gradient of linear part.
        params = [self.W[0], self.b[0]]
        grads = T.grad(cost=None, wrt=params, known_grads=known_grads, 
                        consider_constant=[self.m, self.x],
                        disconnected_inputs='raise')
        cost_grads = T.grad(cost=cost, wrt=params,
                        consider_constant=[self.m, self.x],
                        disconnected_inputs='ignore')
                    
        updates[self.linear_grad] = T.abs_(grads[0]).mean()
        
        for i in range(len(grads)):
            grads[i] += cost_grads[i]
            
        gradients = OrderedDict(izip(params, grads))
        
        # gradients of non-linear part:
        ## start by getting gradients at binary mask:
        params = [self.m]
        grads = T.grad(cost=None, wrt=params, known_grads=known_grads, 
                        consider_constant=[self.m, self.x],
                        disconnected_inputs='raise')
        print "grads at bin", grads
        
        # estimate gradient at simoid input using above:
        grad_m = grads[0]
        if self.derive_sigmoid:
            # multiplying by derivative of sigmoid is optional:
            known_grads[self.a] \
                = grad_m * self.m_mean * (1. - self.m_mean)
        else:
            known_grads[self.a] = grad_m
            
        params = [self.W[1],self.W[2],self.b[1],self.b[2]]
        
        grads = T.grad(cost=None, wrt=params, known_grads=known_grads,
                       consider_constant=[self.z, self.x],
                       disconnected_inputs='raise')
                       
        updates[self.stoch_grad] = T.abs_(grads[1]).mean()
    
        cost_grads = T.grad(cost=cost, wrt=params,
                       consider_constant=[self.z, self.x],
                       disconnected_inputs='ignore')
                       
        updates[self.kl_grad] = T.abs_(cost_grads[1]).mean()
                       
        for i in range(len(grads)):
            grads[i] += cost_grads[i]
                       
        gradients.update(OrderedDict(izip(params, grads)))
        
        return gradients, updates
        
    def get_kl_divergence(self):
        '''
        Minimize KL-divergence of unit binomial distributions with 
        binomial distribution of probability self.sparsity_target.
        This could also be modified to keep a running average of unit 
        samples
        '''
        e = 1e-6
        cost = - self.sparsity_cost_coeff * ( \
                (self.sparsity_target * T.log(e+self.m_mean.mean(axis=0))) \
                +((1.-self.sparsity_target) * T.log(e+(1.-self.m_mean.mean(axis=0)))) \
             ).sum()
        return cost
        
    def get_weight_decay(self):
        rval = 0
        for i in range(3):
            if self.weight_decay_coeff[i] is not None:
                rval += self.weight_decay_coeff[i]*T.sqr(self.W[i]).sum()
        return rval

class CurveThrough(StraightThrough):
    """
    Like Straight-Through (ST) but with the gradient at a (pre-sigmoid)
    multiplied by sqr(b - p) where b is mask, and p is sigmoid.
    
    Or multiplied by Bin(e)/e where e is hyper-parameter between 0 and 1
    and Bin is a binomial distribution.
    """      
    def __init__(self,
                 dim,
                 hidden_dim,
                 layer_name,
                 hidden_activation = 'tanh',
                 expert_activation = 'linear',
                 derive_sigmoid = True,
                 curve = 'sqr(b-p)',
                 curve_noise = None,
                 sparsity_target = 0.1,
                 sparsity_cost_coeff = 1.0,
                 irange = [None,None,None],
                 istdev = [None,None,None],
                 sparse_init = [None,None,None],
                 sparse_stdev = [1.,1.,1.],
                 init_bias = [0.,0.,0.],
                 W_lr_scale = [None,None,None],
                 b_lr_scale = [None,None,None],
                 max_col_norm = [None,None,None],
                 weight_decay_coeff = [None,None,None]):
        '''
        params
        ------
        dim: 
            number of units on output layer
        hidden_dim: 
            number of units on hidden layer of non-linear part
        hidden_activation:
            activation function used on hidden layer of non-linear part
        sparsity_target:
            target sparsity of the output layer.
        sparsity_cost_coeff:
            coefficient of the sparsity constraint when summing costs
        weight_decay_coeff:
            coefficients of L2 weight decay when summing costs
        other:
            in the lists of params, the first index is for the linear 
            part, while the second and third indices are for the first 
            and second layer of the non-linear part, respectively
        
        '''
                     
        self.__dict__.update(locals())
        del self.self
        
    def get_gradients(self, known_grads, loss):
        '''
        Computes gradients and updates for this layer given the known
        gradients of the upper layers, and the vector of losses for the
        batch.
        '''
        updates = OrderedDict()
        
        cost = self.get_kl_divergence() + self.get_weight_decay()
        # gradient of linear part.
        params = [self.W[0], self.b[0]]
        grads = T.grad(cost=None, wrt=params, known_grads=known_grads, 
                        consider_constant=[self.m, self.x],
                        disconnected_inputs='raise')
        cost_grads = T.grad(cost=cost, wrt=params,
                        consider_constant=[self.m, self.x],
                        disconnected_inputs='ignore')
                    
        updates[self.linear_grad] = T.abs_(grads[0]).mean()
        
        for i in range(len(grads)):
            grads[i] += cost_grads[i]
            
        gradients = OrderedDict(izip(params, grads))
        
        # gradients of non-linear part:
        ## start by getting gradients at binary mask:
        params = [self.m]
        grads = T.grad(cost=None, wrt=params, known_grads=known_grads, 
                        consider_constant=[self.m, self.x],
                        disconnected_inputs='raise')
        print "grads at bin", grads
        
        # estimate gradient at simoid input using above:
        grad_a = grads[0]
        if self.derive_sigmoid:
            # multiplying by derivative of sigmoid is optional:
            grad_a *= self.m_mean * (1. - self.m_mean)
           
        if self.curve == 'sqr(b-p)':
            grad_a *= T.sqr(self.m - self.m_mean)
        
        elif self.curve == 'Bin(e)/e':
            assert (self.curve_noise is not None)
            # sample from e to generate a mask
            rng = MRG_RandomStreams(self.mlp.rng.randint(2**15))
            curve_mask = rng.binomial(size = grad_a.shape, n = 1, 
                    p = self.curve_noise, dtype=grad_a.type.dtype)
            grad_a *= curve_mask / self.curve_noise
           
        known_grads[self.a] = grad_m
            
        params = [self.W[1],self.W[2],self.b[1],self.b[2]]
        
        grads = T.grad(cost=None, wrt=params, known_grads=known_grads,
                       consider_constant=[self.z, self.x],
                       disconnected_inputs='raise')
                       
        updates[self.stoch_grad] = T.abs_(grads[1]).mean()
    
        cost_grads = T.grad(cost=cost, wrt=params,
                       consider_constant=[self.z, self.x],
                       disconnected_inputs='ignore')
                       
        updates[self.kl_grad] = T.abs_(cost_grads[1]).mean()
                       
        for i in range(len(grads)):
            grads[i] += cost_grads[i]
                       
        gradients.update(OrderedDict(izip(params, grads)))
        
        return gradients, updates
        
