# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 19:56:19 2013

@author: Nicholas Léonard
"""

import time, sys, math

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


class Conditional1Cost(Default):
    def get_gradients(self, model, data, ** kwargs):
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

        params = list(model.get_params())
        
        layers = model.layers
        
        updates = OrderedDict()
        
        '''Get layer costs'''
        consider_constant = []
        for layer in layers:
            if hasattr(layer,'get_cost'):
                cost += layer.get_cost()
            if hasattr(layer, 'get_consider_constant'):
                consider_constant.extend(layer.get_consider_constant())
            if hasattr(layer, 'get_updates'):
                updates.update(layer.get_updates())
            

        grads = T.grad(cost, params, disconnected_inputs = 'raise',
                       consider_constant = consider_constant)

        gradients = OrderedDict(izip(params, grads))

        return gradients, updates      
            
    def get_test_cost(self, model, X, Y):
        state_below = X
        for layer in model.layers:
            if hasattr(layer, 'test_fprop'):
                state_below = layer.test_fprop(state_below)
            else:
                state_below = layer.fprop(state_below)
        Y_hat = state_below
            
        if self.cost_type == 'default':
            cost = model.cost(Y, Y_hat)
        elif self.cost_type == 'nll':
            cost = (-Y * T.log(Y_hat)).sum(axis=1).mean()
        elif self.cost_type == 'crossentropy':
            cost = (-Y * T.log(Y_hat) - (1 - Y) \
                * T.log(1 - Y_hat)).sum(axis=1).mean()
                
        return cost

class SmoothTimesStochastic(Layer):
    """
    Formerly conditional3
    
    Aaron's gater. Semi-stochastic. Learns through gradient descent.
    
    STS
    
    A linear layer for the main part, 
    and two layers with sigmoid outputs and non-linear hidden units 
    that generates a sparse continuous mask for the outputs of the 
    main part.
    """

    def __init__(self,
                 dim,
                 hidden_dim,
                 layer_name,
                 hidden_activation = 'tanh',
                 sparsity_target = 0.1,
                 sparsity_cost_coeff = 1.0,
                 noise_beta = 1.1,
                 noise_scale = None,
                 noise_stdev = 14.0,
                 noise_normality = 1.0,
                 stochastic_ratio = 0.5,
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
        variance_beta:
            beta coefficient of the beta distribution. The alpha is 
            calculated so that the mode of the distribution is equal to
            the sparsity target. The beta distribution criteria is used
            to encourage variance for each unit.
        variance_cost_coeff:
            coefficient of the variance constraint when summing costs
        weight_decay_coeff:
            coefficients of L2 weight decay when summing costs
        other:
            in the lists of params, the first index is for the linear 
            part, while the second and third indices are for the first 
            and second layer of the non-linear part, respectively
        
        '''
            
        self.__dict__.update(locals())
        del self.self
        
        if self.noise_beta == 0:
            self.noise_beta = None
        print self.noise_beta
        if self.noise_beta is not None:
            if noise_scale is None:
                r = sparsity_target**(1./stochastic_ratio)
                self.noise_scale = abs(-math.log((1./r)-1.))
                print 'noise scale', self.noise_scale
            beta = self.noise_beta
            s = self.sparsity_target
            alpha = (s*(beta-2.0)+1.0)/(1.0-s)
            print 'alpha, beta', alpha, beta
            self.noise_alpha = alpha
            
            self.beta_dist = sharedX((np.random.beta(alpha,beta,size=(3200,dim))-0.5)*self.noise_scale)
            self.beta_idx = theano.shared(int(0))
            self.beta_mean = self.beta_dist.get_value().mean()
            
        elif self.noise_scale is None:
            self.noise_scale = 1.0

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
        rval['mean_output_sparsity'] = self.stoch_m_mean.mean()
        # proportion of sigmoids that have prob > 0.5
        # good when equal to sparsity
        floatX = theano.config.floatX
        rval['mean_sparsity_prop'] \
            = T.cast(T.gt(self.stoch_m_mean, 0.5),floatX).mean()
        # same as above but for intermediate thresholds:
        rval['mean_sparsity_prop0.2'] \
            = T.cast(T.gt(self.stoch_m_mean, 0.2),floatX).mean()
        rval['mean_sparsity_prop0.3'] \
            = T.cast(T.gt(self.stoch_m_mean, 0.3),floatX).mean()
        rval['mean_sparsity_prop0.4'] \
            = T.cast(T.gt(self.stoch_m_mean, 0.4),floatX).mean()    
        # or just plain standard deviation (less is bad):
        rval['output_stdev'] = self.stoch_m_mean.std()
        # stdev of unit stdevs (more is bad)
        rval['output_meta_stdev'] = self.stoch_m_mean.std(axis=0).std()
        # max and min proportion of these probs per unit
        prop_per_unit = T.cast(T.gt(self.stoch_m_mean, 0.5),floatX).mean(0)
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

    def fprop(self, state_below, add_noise=True, threshold=None, stochastic=True):
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
            self.z = S.dot(self.x,self.W[0]) + self.b[0]
        else:
            self.z = T.dot(self.x,self.W[0]) + self.b[0]


        self.stopper = self.x * T.ones_like(self.x)
        # first layer non-linear part
        if isinstance(self.stopper, S.SparseVariable):
            h = S.dot(self.stopper,self.W[1]) + self.b[1]
        else:
            h = T.dot(self.stopper,self.W[1]) + self.b[1]
        
        # activate hidden units of non-linear part
        if self.hidden_activation is None:
            pass
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
        
        
        rng = MRG_RandomStreams(self.mlp.rng.randint(2**15))
        noise = 0
        if self.noise_beta is not None:
            noise = (1.-self.noise_normality) * self.beta_mean
        print noise
        
        if add_noise:
            if self.noise_beta is not None:
                noise = (1.-self.noise_normality) * self.beta_dist[ \
                        self.beta_idx:self.beta_idx+self.x.shape[0],:] \
                        + (self.noise_normality * self.noise_scale \
                                * rng.normal(size = self.z.shape, 
                                        std=self.noise_stdev ,
                                        dtype=self.z.type.dtype) \
                            )
            else:
                noise = self.noise_scale \
                    * rng.normal(size = self.z.shape, 
                                        std=self.noise_stdev ,
                                        dtype=self.z.type.dtype)
        
        
        # second layer non-linear part
        self.a = T.dot(self.h,self.W[2]) + self.b[2] + noise
        
        # activate non-linear part to get bernouilli probabilities
        self.m_mean = T.nnet.sigmoid(self.a)
        
        # Separate stochastic from deterministic part:
        self.stoch_m_mean = self.m_mean**self.stochastic_ratio
        self.deter_m_mean = self.m_mean**(1.-self.stochastic_ratio)
        
        if threshold is None:
            if stochastic:
                # sample from bernouili probs to generate a mask
                self.m = rng.binomial(size = self.stoch_m_mean.shape, n = 1 , \
                    p = self.m_mean, dtype=self.stoch_m_mean.type.dtype)
            else:
                self.m = self.m_mean
        else:
            # deterministic mask:
            self.m = T.cast(T.gt(self.stoch_m_mean, threshold), \
                                           theano.config.floatX)
                                        
        self.consider_constant = [self.m, self.stopper]
           
        # mix output of linear part with output of non-linear part
        self.p = self.m * self.deter_m_mean * self.z
        
        if self.layer_name is not None:
            self.z.name = self.layer_name + '_z'
            self.h.name = self.layer_name + '_h'
            self.a.name = self.layer_name + '_a'
            self.m_mean.name = self.layer_name + '_m_mean'
            self.stoch_m_mean.name = self.layer_name + '_stoch_m_mean'
            self.deter_m_mean.name = self.layer_name + '_deter_m_mean'
            self.p.name = self.layer_name + '_p'
        
        return self.p

    def test_fprop(self, state_below, threshold=None, stochastic=True):
        return self.fprop(state_below, add_noise=False, threshold=threshold, stochastic=stochastic)

    def cost(self, Y, Y_hat):
        return self.cost_from_cost_matrix(self.cost_matrix(Y, Y_hat))

    def cost_from_cost_matrix(self, cost_matrix):
        return cost_matrix.sum(axis=1).mean()

    def cost_matrix(self, Y, Y_hat):
        return T.sqr(Y - Y_hat)
        
    def get_consider_constant(self):
        '''
        T.grad complains when trying to propagate gradients through
        random distribution functions like binomial.
        '''
        return self.consider_constant
        
    def get_cost(self):
        return self.get_kl_divergence() + self.get_weight_decay()
        
    def get_kl_divergence(self):
        '''
        Minimize KL-divergence of unit binomial distributions with 
        binomial distribution of probability self.sparsity_target.
        This could also be modified to keep a running average of unit 
        samples
        '''
        e = 1e-6
        cost = - self.sparsity_cost_coeff * ( \
                (self.sparsity_target * T.log(e+self.stoch_m_mean.mean(axis=0))) \
                +((1.-self.sparsity_target) * T.log(e+(1.-self.stoch_m_mean.mean(axis=0)))) \
             ).sum()
        return cost
        
    def get_weight_decay(self):
        rval = 0
        for i in range(3):
            if self.weight_decay_coeff[i] is not None:
                rval += self.weight_decay_coeff[i]*T.sqr(self.W[i]).sum()
        return rval
        
    def get_updates(self):
        if self.noise_beta is not None:
            return {self.beta_idx: (self.beta_idx + 32) % 3200}
        return {}
        
        
 
class NoisyRectifier(Layer):
    """
    Formerly Conditional4
    
    Yoshua's Semi-hard Stochastic ReLU 
    A linear layer for the main part, 
    and two layers with sigmoid outputs and non-linear hidden units 
    that generates a sparse continuous mask for the outputs of the 
    main part.
    """

    def __init__(self,
                 dim,
                 hidden_dim,
                 layer_name,
                 hidden_activation = 'tanh',
                 noise_stdev = 14.0,
                 sparsity_cost_coeff = 0.001,
                 sparsity_target = 0.1,
                 sparsity_decay = 0.95,
                 gater_activation = 'rectifiedlinear',
                 expert_activation = 'linear',
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
        sparsity_cost_coeff:
            coefficient of the sparsity constraint when summing costs
        gater_activation:
            activation function used on the output of the gater
        weight_decay_coeff:
            coefficients of L2 weight decay when summing costs
        other:
            in the lists of params, the first index is for the linear 
            part, while the second and third indices are for the first 
            and second layer of the non-linear part, respectively
        
        '''
                     
        self.__dict__.update(locals())
        del self.self
        
        self.sparsity_cost_coeff = sharedX(sparsity_cost_coeff)
        self.max_sparsity_cc = sparsity_cost_coeff

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
        for i in range(3):
            sq_W = T.sqr(self.W[i])

            row_norms = T.sqrt(sq_W.sum(axis=1))
            col_norms = T.sqrt(sq_W.sum(axis=0))
            
            rval['row_norms_max'+str(i)] = row_norms.max()
            rval['col_norms_max'+str(i)] = col_norms.max()
        
        return rval
        
    def get_monitoring_channels_from_state(self, state, target=None):
        rval =  OrderedDict()
        rval['sparsity_cost_coeff'] = self.sparsity_cost_coeff
        # sparisty of outputs:
        rval['mean_output_sparsity'] = self.m_mean.mean()
        # proportion of sigmoids that have prob > 0.5
        # good when equal to sparsity
        floatX = theano.config.floatX
        rval['mean_sparsity_prop'] \
            = self.effective_sparsity
        # same as above but for intermediate thresholds:
        rval['mean_sparsity_prop0.2'] \
            = T.cast(T.gt(self.m_mean, 0.2),floatX).mean()
        rval['mean_sparsity_prop0.3'] \
            = T.cast(T.gt(self.m_mean, 0.3),floatX).mean()
        rval['mean_sparsity_prop0.4'] \
            = T.cast(T.gt(self.m_mean, 0.4),floatX).mean() 
        rval['mean_a'] = self.a.mean()
        rval['stdev_a'] = self.a.std()
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

    def fprop(self, state_below, add_noise=True):
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
        
        # activate hidden units of non-linear part
        self.h = self.activate(h, self.hidden_activation)
            
        noise = 0.
        if add_noise:
            rng = MRG_RandomStreams(self.mlp.rng.randint(2**15))
            noise = rng.normal(size = self.z.shape, 
                                    std=self.noise_stdev ,
                                    dtype=self.z.type.dtype) 
        
        # second layer non-linear part
        self.a = T.dot(self.h,self.W[2]) + self.b[2] + noise
        
        # activate non-linear part
        self.m_mean = self.activate(self.a, self.gater_activation)
        
        # how many are over 0:
        self.effective_sparsity = T.cast(T.gt(self.m_mean, 0), 
                                         theano.config.floatX).mean()
           
        # mix output of linear part with output of non-linear part
        self.p = self.m_mean * self.z
        
        if self.layer_name is not None:
            self.z.name = self.layer_name + '_z'
            self.h.name = self.layer_name + '_h'
            self.a.name = self.layer_name + '_a'
            self.m_mean.name = self.layer_name + '_m_mean'
            self.p.name = self.layer_name + '_p'
        
        return self.p
        
    def test_fprop(self, state_below):
        return self.fprop(state_below, add_noise=False)

    def cost(self, Y, Y_hat):
        return self.cost_from_cost_matrix(self.cost_matrix(Y, Y_hat))

    def cost_from_cost_matrix(self, cost_matrix):
        return cost_matrix.sum(axis=1).mean()

    def cost_matrix(self, Y, Y_hat):
        return T.sqr(Y - Y_hat)
        
    def get_cost(self):
        return self.get_l1_norm() + self.get_weight_decay() 
    
    def get_updates(self):
        """ Keeps the effective sparsity around sparsity_target """
        updates = OrderedDict()
        updates[self.sparsity_cost_coeff] \
            = T.clip( \
                self.sparsity_cost_coeff - \
                (T.cast(T.gt( \
                    self.sparsity_target-self.effective_sparsity,0.01),
                    theano.config.floatX \
                ) * self.sparsity_decay * self.sparsity_cost_coeff) \
                + (T.cast(T.lt( \
                    self.sparsity_target-self.effective_sparsity,-0.01),
                    theano.config.floatX \
                ) * self.sparsity_decay * self.sparsity_cost_coeff) \
            ,0.0000000000001,1)
        return updates
        
    def get_l1_norm(self):
        '''
        Minimize l1 norm of hiddens
        '''
        return self.sparsity_cost_coeff * abs(self.m_mean).sum()
        
    def get_weight_decay(self):
        rval = 0
        for i in range(3):
            if self.weight_decay_coeff[i] is not None:
                rval += self.weight_decay_coeff[i]*T.sqr(self.W[i]).sum()
        return rval
        

class BaselineSigmoid(Layer):
    """
    Formerly Conditional5
    
    A linear layer for the main part, 
    and two layers with sigmoid outputs and non-linear hidden units 
    that generates a sparse continuous mask for the outputs of the 
    main part.
    """

    def __init__(self,
                 dim,
                 hidden_dim,
                 layer_name,
                 hidden_activation = 'tanh',
                 sparsity_target = 0.1,
                 sparsity_cost_coeff = 1.0,
                 noise_beta = 1.1,
                 noise_scale = 2.1976,
                 noise_stdev = 14.0,
                 noise_normality = 0.5,
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
        variance_beta:
            beta coefficient of the beta distribution. The alpha is 
            calculated so that the mode of the distribution is equal to
            the sparsity target. The beta distribution criteria is used
            to encourage variance for each unit.
        variance_cost_coeff:
            coefficient of the variance constraint when summing costs
        weight_decay_coeff:
            coefficients of L2 weight decay when summing costs
        other:
            in the lists of params, the first index is for the linear 
            part, while the second and third indices are for the first 
            and second layer of the non-linear part, respectively
        
        '''
                     
        self.__dict__.update(locals())
        del self.self
        if self.noise_beta == 0:
            self.noise_beta = None
        print self.noise_beta
        if self.noise_beta is not None:
            beta = self.noise_beta
            s = self.sparsity_target
            alpha = (s*(beta-2.0)+1.0)/(1.0-s)
            print 'alpha, beta', alpha, beta
            self.noise_alpha = alpha
            
            self.beta_dist = sharedX((np.random.beta(alpha,beta,size=(3200,dim))-0.5)*self.noise_scale)
            self.beta_idx = theano.shared(int(0))
            self.beta_mean = self.beta_dist.get_value().mean()


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

    def fprop(self, state_below, add_noise=True):
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
            self.z = S.dot(self.x,self.W[0]) + self.b[0]
        else:
            self.z = T.dot(self.x,self.W[0]) + self.b[0]
        
        # first layer non-linear part
        if isinstance(self.x, S.SparseVariable):
            h = S.dot(self.x,self.W[1]) + self.b[1]
        else:
            h = T.dot(self.x,self.W[1]) + self.b[1]
        
        # activate hidden units of non-linear part
        if self.hidden_activation is None:
            pass
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
        
        noise = 0
        if self.noise_beta is not None:
            noise = (1.-self.noise_normality) * self.beta_mean
            #print self.noise_normality
            print (1.-self.noise_normality) * self.noise_scale * (self.sparsity_target - 0.5)
        print noise
        
        if add_noise:
            rng = MRG_RandomStreams(self.mlp.rng.randint(2**15))
            if self.noise_beta is not None:
                noise = (1.-self.noise_normality) * self.beta_dist[ \
                        self.beta_idx:self.beta_idx+self.x.shape[0],:] \
                        + (self.noise_normality * self.noise_scale \
                                * rng.normal(size = self.z.shape, 
                                        std=self.noise_stdev ,
                                        dtype=self.z.type.dtype) \
                            )
            else:
                noise = self.noise_scale \
                    * rng.normal(size = self.z.shape, 
                                        std=self.noise_stdev ,
                                        dtype=self.z.type.dtype)
                
        #print self.beta_dist.get_value().shape
            
        # second layer non-linear part
        self.a = T.dot(self.h,self.W[2]) + self.b[2] + noise
        
        # activate non-linear part to get bernouilli probabilities
        self.m_mean = T.nnet.sigmoid(self.a)
           
        # mix output of linear part with output of non-linear part
        self.p = self.m_mean * self.z
        
        if self.layer_name is not None:
            self.z.name = self.layer_name + '_z'
            self.h.name = self.layer_name + '_h'
            self.a.name = self.layer_name + '_a'
            self.m_mean.name = self.layer_name + '_m_mean'
            self.p.name = self.layer_name + '_p'
        
        return self.p
        
    def test_fprop(self, state_below):
        return self.fprop(state_below, add_noise=False)

    def cost(self, Y, Y_hat):
        return self.cost_from_cost_matrix(self.cost_matrix(Y, Y_hat))

    def cost_from_cost_matrix(self, cost_matrix):
        return cost_matrix.sum(axis=1).mean()

    def cost_matrix(self, Y, Y_hat):
        return T.sqr(Y - Y_hat)
        
    def get_cost(self):
        return self.get_kl_divergence() + self.get_weight_decay()
        
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
        
    def get_updates(self):
        if self.noise_beta is not None:
            return {self.beta_idx: (self.beta_idx + 32) % 3200}
        return {}

