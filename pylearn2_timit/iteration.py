import numpy
from theano import config
from pylearn2.space import CompositeSpace
from pylearn2.utils import safe_zip
from pylearn2.utils.data_specs import is_flat_specs


class FiniteDatasetIterator(object):
    """
    A thin wrapper around one of the mode iterators.
    """
    def __init__(self, dataset, subset_iterator,
                 data_specs=None, return_tuple=False, convert=None):
        """
        .. todo::

            WRITEME
        """

        self._data_specs = data_specs
        self._dataset = dataset
        self._subset_iterator = subset_iterator
        self._return_tuple = return_tuple

        assert is_flat_specs(data_specs)

        dataset_space, dataset_source = self._dataset.get_data_specs()
        assert is_flat_specs((dataset_space, dataset_source))

        # the dataset's data spec is either a single (space, source) pair,
        # or a pair of (non-nested CompositeSpace, non-nested tuple).
        # We could build a mapping and call flatten(..., return_tuple=True)
        # but simply putting spaces, sources and data in tuples is simpler.
        if not isinstance(dataset_source, tuple):
            dataset_source = (dataset_source,)

        if not isinstance(dataset_space, CompositeSpace):
            dataset_sub_spaces = (dataset_space,)
        else:
            dataset_sub_spaces = dataset_space.components
        assert len(dataset_source) == len(dataset_sub_spaces)

        space, source = data_specs
        if not isinstance(source, tuple):
            source = (source,)
        if not isinstance(space, CompositeSpace):
            sub_spaces = (space,)
        else:
            sub_spaces = space.components
        assert len(source) == len(sub_spaces)

        self._source = source

        if convert is None:
            self._convert = [None for s in source]
        else:
            assert len(convert) == len(source)
            self._convert = convert

        dtypes = self._dataset.dtype_of(self._source)
        for i, (so, sp, dt) in enumerate(safe_zip(source, sub_spaces, dtypes)):
            idx = dataset_source.index(so)
            dspace = dataset_sub_spaces[idx]

            init_fn = self._convert[i]
            fn = init_fn
            # Compose the functions
            needs_cast = not (numpy.dtype(config.floatX) == dt)
            if needs_cast:
                if fn is None:
                    fn = lambda batch: numpy.cast[config.floatX](batch)
                else:
                    fn = (lambda batch, fn_=fn:
                          numpy.cast[config.floatX](fn_(batch)))

            # If there is an init_fn, it is supposed to take
            # care of the formatting, and it should be an error
            # if it does not. If there was no init_fn, then
            # the iterator will try to format using the generic
            # space-formatting functions.
            needs_format = not init_fn and not sp == dspace
            if needs_format:
                # "dspace" and "sp" have to be passed as parameters
                # to lambda, in order to capture their current value,
                # otherwise they would change in the next iteration
                # of the loop.
                if fn is None:
                    fn = (lambda batch, dspace=dspace, sp=sp:
                          dspace.np_format_as(batch, sp))
                else:
                    fn = (lambda batch, dspace=dspace, sp=sp, fn_=fn:
                          dspace.np_format_as(fn_(batch), sp))

            self._convert[i] = fn

    def __iter__(self):
        """
        .. todo::

            WRITEME
        """
        return self

    def next(self):
        """
        .. todo::

            WRITEME
        """
        next_index = self._subset_iterator.next()
        # TODO: handle fancy-index copies by allocating a buffer and
        # using numpy.take()
        rval = tuple(
            fn(batch) if fn else batch for batch, fn in
            safe_zip(self._dataset.get(self._source, next_index),
                     self._convert)
        )

        if not self._return_tuple and len(rval) == 1:
            rval, = rval
        return rval

    @property
    def batch_size(self):
        """
        .. todo::

            WRITEME
        """
        return self._subset_iterator.batch_size

    @property
    def num_batches(self):
        """
        .. todo::

            WRITEME
        """
        return self._subset_iterator.num_batches

    @property
    def num_examples(self):
        """
        .. todo::

            WRITEME
        """
        return self._subset_iterator.num_examples

    @property
    def uneven(self):
        """
        .. todo::

            WRITEME
        """
        return self._subset_iterator.uneven

    @property
    def stochastic(self):
        """
        .. todo::

            WRITEME
        """
        return self._subset_iterator.stochastic
