# This file shows how to train a binary RBM on MNIST by viewing it as a single layer DBM.
# The hyperparameters in this file aren't especially great; they're mostly chosen to demonstrate
# the interface. Feel free to suggest better hyperparameters!
!obj:pylearn2.train.Train {
    # For this example, we will train on a binarized version of MNIST.
    # We binarize by drawing samples--if an MNIST pixel is set to 0.9,
    # we make binary pixel that is 1 with probability 0.9. We redo the
    # sampling every time the example is presented to the learning
    # algorithm.
    # In pylearn2, we do this by making a Binarizer dataset. The Binarizer
    # is a dataset that can draw samples like this in terms of any
    # input dataset with values in [0,1].
    dataset: &data !obj:timit_dataset.TimitFrameData {
        datapath: '/home/jfsantos/data/TIMIT/',
        framelen: 160,
        overlap: 159,
        start: 0,
        stop: 10
    },
    model: !obj:pylearn2.models.dbm.DBM {
        batch_size: 100,
        # 1 mean field iteration reaches convergence in the RBM
        niter: 1,
        # The visible layer of this RBM is just a binary vector
        # (as opposed to a binary image for convolutional models,
        # a Gaussian distributed vector, etc.)
        visible_layer: !obj:pylearn2.models.dbm.GaussianVisLayer {
            rows: 160,
            cols: 1,
            channels: 1
        },
        hidden_layers: [
            !obj:pylearn2.models.dbm.Softmax {
                layer_name: 'h'
            }
       ]
    },
    # We train the model using stochastic gradient descent.
    # One benefit of using pylearn2 is that we can use the exact same piece of
    # code to train a DBM as to train an MLP. The interface that SGD uses to get
    # the gradient of the cost function from an MLP can also get the *approximate*
    # gradient from a DBM.
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
               # We initialize the learning rate and momentum here. Down below
               # we can control the way they decay with various callbacks.
               learning_rate: 1e-3,
               # Compute new model parameters using SGD + Momentum
               learning_rule: !obj:pylearn2.training_algorithms.learning_rule.Momentum {
                   init_momentum: 0.5,
               },
               # These arguments say to compute the monitoring channels on 10 batches
               # of the training set.
               monitoring_batches: 10,
               monitoring_dataset : *data,
               # The SumOfCosts allows us to add together a few terms to make a complicated
               # cost function.
               cost : !obj:pylearn2.costs.cost.SumOfCosts {
                costs: [
                        # The first term of our cost function is variational PCD.
                        # For the RBM, the variational approximation is exact, so
                        # this is really just PCD. In deeper models, it means we
                        # use mean field rather than Gibbs sampling in the positive phase.
                        !obj:pylearn2.costs.dbm.VariationalPCD {
                           # Here we specify how many fantasy particles to maintain
                           num_chains: 100,
                           # Here we specify how many steps of Gibbs sampling to do between
                           # each parameter update.
                           num_gibbs_steps: 5
                        },
                        # The second term of our cost function is a little bit of weight
                        # decay.
                        !obj:pylearn2.costs.dbm.WeightDecay {
                          coeffs: [ .0001  ]
                        },
                        # Finally, we regularize the RBM to sparse, using a method copied
                        # from Ruslan Salakhutdinov's DBM demo
                        !obj:pylearn2.costs.dbm.TorontoSparsity {
                         targets: [ .2 ],
                         coeffs: [ .001 ],
                        }
                       ],
           },
           # We tell the RBM to train for 300 epochs
           termination_criterion: !obj:pylearn2.termination_criteria.EpochCounter { max_epochs: 300 },
           update_callbacks: [
                # This callback makes the learning rate shrink by dividing it by decay_factor after
                # each sgd step.
                !obj:pylearn2.training_algorithms.sgd.ExponentialDecay {
                        decay_factor: 1.000015,
                        min_lr:       0.0001
                }
           ]
        },
    extensions: [
            # This callback makes the momentum grow to 0.9 linearly. It starts
            # growing at epoch 5 and finishes growing at epoch 6.
            !obj:pylearn2.training_algorithms.learning_rule.MomentumAdjustor {
                final_momentum: .9,
                start: 5,
                saturate: 6
            },
    ],
    # This saves to save the trained model to the same path as
    # the yaml file, but replacing .yaml with .pkl
    save_path: "${PYLEARN2_TRAIN_FILE_FULL_STEM}.pkl",
    # This says to save it every epoch
    save_freq : 1
}

