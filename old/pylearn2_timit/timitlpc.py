"""
Pylearn2 wrapper for the TIMIT dataset
"""
__authors__ = ["Vincent Dumoulin"]
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Laurent Dinh", "Vincent Dumoulin"]
__license__ = "3-clause BSD"
__maintainer__ = "Vincent Dumoulin"
__email__ = "dumouliv@iro"


import gc
import os.path
import functools
import numpy
from pylearn2.utils.iteration import FiniteDatasetIterator, resolve_iterator_class
from pylearn2.datasets.dataset import Dataset
from pylearn2.space import CompositeSpace, VectorSpace, IndexSpace
from pylearn2.utils import serial
from pylearn2.utils import safe_zip
from segmentaxis import segment_axis
import scipy.stats
from scikits.talkbox import lpc

class TIMITlpc(Dataset):
    """
    Frame-based TIMIT dataset
    """
    _default_seed = (17, 2, 946)
    
    # Mean and standard deviation of the acoustic samples from the whole
    # dataset (train, valid, test).
    _mean = 0.0035805809921434142
    _std = 542.48824133746177

    def __init__(self, which_set, frame_length, overlap=0, lpc_order=10,
                 frames_per_example=1, start=0, stop=None, audio_only=False,
                 rng=_default_seed):
        """
        Parameters
        ----------
        which_set : str
            Either "train", "valid" or "test"
        frame_length : int
            Number of acoustic samples contained in a frame
        overlap : int, optional
            Number of overlapping acoustic samples for two consecutive frames.
            Defaults to 0, meaning frames don't overlap.
        frames_per_example : int, optional
            Number of frames in a training example. Defaults to 1.
        start : int, optional
            Starting index of the sequences to use. Defaults to 0.
        stop : int, optional
            Ending index of the sequences to use. Defaults to `None`, meaning
            sequences are selected all the way to the end of the array.
        audio_only : bool, optional
            Whether to load only the raw audio and no auxiliary information.
            Defaults to `False`.
        rng : object, optional
            A random number generator used for picking random indices into the
            design matrix when choosing minibatches.
        """
        self.frame_length = frame_length
        self.overlap = overlap
        self.frames_per_example = frames_per_example
        self.lpc_order = lpc_order
        self.offset = self.frame_length - self.overlap
        self.audio_only = audio_only

        # RNG initialization
        if hasattr(rng, 'random_integers'):
            self.rng = rng
        else:
            self.rng = numpy.random.RandomState(rng)

        # Load data from disk
        self._load_data(which_set)
        # Standardize data
        # for i, sequence in enumerate(self.raw_wav):
        #     self.raw_wav[i] = (sequence - TIMIT._mean) / TIMIT._std

        # Slice data
        if stop is not None:
            self.raw_wav = self.raw_wav[start:stop]
            if not self.audio_only:
                self.phones = self.phones[start:stop]
                self.phonemes = self.phonemes[start:stop]
                self.words = self.words[start:stop]
        else:
            self.raw_wav = self.raw_wav[start:]
            if not self.audio_only:
                self.phones = self.phones[start:]
                self.phonemes = self.phonemes[start:]
                self.words = self.words[start:]

        examples_per_sequence = [0]

        for sequence_id, samples_sequence in enumerate(self.raw_wav):
            if not self.audio_only:
                # Phones segmentation
                phones_sequence = self.phones[sequence_id]
                phones_segmented_sequence = segment_axis(phones_sequence,
                                                         frame_length,
                                                         overlap)
                self.phones[sequence_id] = phones_segmented_sequence
                # phones_segmented_sequence = scipy.stats.mode(
                #     phones_segmented_sequence,
                #     axis=1
                # )[0].flatten()
                # phones_segmented_sequence = numpy.asarray(
                #     phones_segmented_sequence,
                #     dtype='int'
                # )
                # phones_sequence_list.append(phones_segmented_sequence)
                # Phonemes segmentation
                phonemes_sequence = self.phonemes[sequence_id]
                phonemes_segmented_sequence = segment_axis(phonemes_sequence,
                                                           frame_length,
                                                           overlap)
                self.phonemes[sequence_id] = phonemes_segmented_sequence
                # phonemes_segmented_sequence = scipy.stats.mode(
                #     phonemes_segmented_sequence,
                #     axis=1
                # )[0].flatten()
                # phonemes_segmented_sequence = numpy.asarray(
                #     phonemes_segmented_sequence,
                #     dtype='int'
                # )
                # phonemes_sequence_list.append(phonemes_segmented_sequence)
                # Words segmentation
                words_sequence = self.words[sequence_id]
                words_segmented_sequence = segment_axis(words_sequence,
                                                        frame_length,
                                                        overlap)
                self.words[sequence_id] = words_segmented_sequence
                # words_segmented_sequence = scipy.stats.mode(
                #     words_segmented_sequence,
                #     axis=1
                # )[0].flatten()
                # words_segmented_sequence = numpy.asarray(words_segmented_sequence,
                #                                          dtype='int')
                # words_sequence_list.append(words_segmented_sequence)

            # TODO: look at this, does it force copying the data?
            # Sequence segmentation
            samples_segmented_sequence = segment_axis(samples_sequence,
                                                      frame_length,
                                                      overlap)
            self.raw_wav[sequence_id] = samples_segmented_sequence

            # TODO: change me
            # Generate features/targets/phones/phonemes/words map
            num_frames = samples_segmented_sequence.shape[0]
            num_examples = num_frames - self.frames_per_example
            examples_per_sequence.append(num_examples)

        self.cumulative_example_indexes = numpy.cumsum(examples_per_sequence)
        self.samples_sequences = self.raw_wav
        if not self.audio_only:
            self.phones_sequences = self.phones
            self.phonemes_sequences = self.phonemes
            self.words_sequences = self.words
        self.num_examples = self.cumulative_example_indexes[-1]

        # DataSpecs
        # features_space = VectorSpace(
        #     dim=self.frame_length * self.frames_per_example
        # )
        # features_source = 'features'
        # features_dtype = self.samples_sequences[0].dtype
        # def features_map_fn(indexes):
        #     rval = []
        #     for sequence_index, example_index in self._fetch_index(indexes):
        #         rval.append(self.samples_sequences[sequence_index][example_index:example_index
        #             + self.frames_per_example].ravel())
        #     return rval

        targets_space = VectorSpace(
            dim=self.lpc_order
        )
        targets_source = 'targets'
        targets_dtype = numpy.float64
        def targets_map_fn(indexes):
            rval = []
            for sequence_index, example_index in self._fetch_index(indexes):
                rval.append(lpc(self.samples_sequences[sequence_index][example_index].T,self.lpc_order)[0][1:].ravel())
            return rval

        features_space = IndexSpace(max_labels=62, dim=3)
        features_source = 'features'
        features_dtype = numpy.int16
        def features_map_fn(indexes):
            rval = []
            for sequence_index, example_index in self._fetch_index(indexes):
                if example_index != 0:
                    if example_index != self.num_examples-1:
                        rval.append(numpy.asarray(scipy.stats.mode(self.phones_sequences[sequence_index][(example_index-1):(example_index+2)], axis=1)[0].T, dtype=numpy.int16))
                    else:
                        rval.append(numpy.asarray(numpy.concatenate([scipy.stats.mode(self.phones_sequences[sequence_index][example_index-1:example_index+1], axis=1)[0].T, numpy.zeros((1,1))], axis=1).ravel(), dtype=numpy.int16))
                else:
                    rval.append(numpy.asarray(numpy.concatenate([numpy.zeros((1,1)), scipy.stats.mode(self.phones_sequences[sequence_index][example_index:example_index+2].T)[0].T]).ravel(), dtype=numpy.int16))
            return rval

        # targets_space = VectorSpace(dim=self.frame_length)
        # targets_source = 'targets'
        # targets_dtype = self.samples_sequences[0].dtype
        # def targets_map_fn(indexes):
        #     rval = []
        #     for sequence_index, example_index in self._fetch_index(indexes):
        #         rval.append(self.samples_sequences[sequence_index][example_index
        #             + self.frames_per_example].ravel())
        #     return rval

        space_components = [features_space, targets_space]
        source_components = [features_source, targets_source]
        dtypes_components = [features_dtype, targets_dtype]
        map_fn_components = [features_map_fn, targets_map_fn]
        batch_components = [None, None]

        # if not self.audio_only:
        #     phones_space = IndexSpace(max_labels=62, dim=3)
        #     phones_source = 'phones'
        #     phones_dtype = self.phones_sequences[0].dtype
        #     def phones_map_fn(indexes):
        #         rval = []
        #         for sequence_index, example_index in self._fetch_index(indexes):
        #             if example_index != 0:
        #                 if example_index != self.num_examples-1:
        #                     rval.append(scipy.stats.mode(self.phones_sequences[sequence_index][(example_index-1):(example_index+2)], axis=1)[0].T)
        #                 else:
        #                     rval.append(numpy.concatenate([scipy.stats.mode(self.phones_sequences[sequence_index][example_index-1:example_index+1], axis=1)[0].T, numpy.zeros((1,1))], axis=1)).ravel()
        #             else:
        #                 rval.append(numpy.concatenate([numpy.zeros((1,1)), scipy.stats.mode(self.phones_sequences[sequence_index][example_index:example_index+1])[0].T])).ravel()
        #         return rval

        #     phonemes_space = IndexSpace(max_labels=31, dim=1)
        #     phonemes_source = 'phonemes'
        #     phonemes_dtype = self.phonemes_sequences[0].dtype
        #     def phonemes_map_fn(indexes):
        #         rval = []
        #         for sequence_index, example_index in self._fetch_index(indexes):
        #             rval.append(self.phonemes_sequences[sequence_index][example_index
        #                 + self.frames_per_example].ravel())
        #         return rval

        #     words_space = IndexSpace(max_labels=31, dim=1)
        #     words_source = 'words'
        #     words_dtype = self.words_sequences[0].dtype
        #     def words_map_fn(indexes):
        #         rval = []
        #         for sequence_index, example_index in self._fetch_index(indexes):
        #             rval.append(self.words_sequences[sequence_index][example_index
        #                 + self.frames_per_example].ravel())
        #         return rval

        #     space_components.extend([phones_space, phonemes_space,
        #                              words_space])
        #     source_components.extend([phones_source, phonemes_source,
        #                              words_source])
        #     dtypes_components.extend([phones_dtype, phonemes_dtype,
        #                              words_dtype])
        #     map_fn_components.extend([phones_map_fn, phonemes_map_fn,
        #                              words_map_fn])
        #     batch_components.extend([None, None, None])

        space = CompositeSpace(space_components)
        source = tuple(source_components)
        self.data_specs = (space, source)
        self.dtypes = tuple(dtypes_components)
        self.map_functions = tuple(map_fn_components)
        self.batch_buffers = batch_components

        # Defaults for iterators
        self._iter_mode = resolve_iterator_class('shuffled_sequential')
        self._iter_data_specs = (CompositeSpace((features_space,
                                                 targets_space)),
                                 (features_source, targets_source))


    def _fetch_index(self, indexes):
        digit = numpy.digitize(indexes, self.cumulative_example_indexes) - 1
        return zip(digit,
                   numpy.array(indexes) - self.cumulative_example_indexes[digit])

    def _load_data(self, which_set):
        """
        Load the TIMIT data from disk.

        Parameters
        ----------
        which_set : str
            Subset of the dataset to use (either "train", "valid" or "test")
        """
        # Check which_set
        if which_set not in ['train', 'valid', 'test']:
            raise ValueError(which_set + " is not a recognized value. " +
                             "Valid values are ['train', 'valid', 'test'].")

        # Create file paths
        timit_base_path = os.path.join(os.environ["PYLEARN2_DATA_PATH"],
                                       "timit/readable")
        speaker_info_list_path = os.path.join(timit_base_path, "spkrinfo.npy")
        phonemes_list_path = os.path.join(timit_base_path,
                                          "reduced_phonemes.pkl")
        words_list_path = os.path.join(timit_base_path, "words.pkl")
        speaker_features_list_path = os.path.join(timit_base_path,
                                                  "spkr_feature_names.pkl")
        speaker_id_list_path = os.path.join(timit_base_path,
                                            "speakers_ids.pkl")
        raw_wav_path = os.path.join(timit_base_path, which_set + "_x_raw.npy")
        phonemes_path = os.path.join(timit_base_path,
                                     which_set + "_x_phonemes.npy")
        phones_path = os.path.join(timit_base_path,
                                     which_set + "_x_phones.npy")
        words_path = os.path.join(timit_base_path, which_set + "_x_words.npy")
        speaker_path = os.path.join(timit_base_path,
                                    which_set + "_spkr.npy")

        # Load data. For now most of it is not used, as only the acoustic
        # samples are provided, but this is bound to change eventually.
        # Global data
        if not self.audio_only:
            self.speaker_info_list = serial.load(
                speaker_info_list_path
            ).tolist().toarray()
            self.speaker_id_list = serial.load(speaker_id_list_path)
            self.speaker_features_list = serial.load(speaker_features_list_path)
            self.words_list = serial.load(words_list_path)
            self.phonemes_list = serial.load(phonemes_list_path)
        # Set-related data
        self.raw_wav = serial.load(raw_wav_path)
        if not self.audio_only:
            self.phonemes = serial.load(phonemes_path)
            self.phones = serial.load(phones_path)
            self.words = serial.load(words_path)
            self.speaker_id = numpy.asarray(serial.load(speaker_path), 'int')

    def _validate_source(self, source):
        """
        Verify that all sources in the source tuple are provided by the
        dataset. Raise an error if some requested source is not available.

        Parameters
        ----------
        source : `tuple` of `str`
            Requested sources
        """
        for s in source:
            try:
                self.data_specs[1].index(s)
            except ValueError:
                raise ValueError("the requested source named '" + s + "' " +
                                 "is not provided by the dataset")

    def dtype_of(self, source):
        """
        Returns the dtype of the requested source
        """
        self._validate_source(source)
        return tuple(self.dtypes[self.data_specs[1].index(so)]
                     for so in source)

    def get_data_specs(self):
        """
        Returns the data_specs specifying how the data is internally stored.

        This is the format the data returned by `self.get_data()` will be.

        .. note::

            Once again, this is very hacky, as the data is not stored that way
            internally. However, the data that's returned by `TIMIT.get()`
            _does_ respect those data specs.
        """
        return self.data_specs

    def get(self, source, indexes):
        """
        .. todo::

            WRITEME
        """
        if type(indexes) is slice:
            indexes = numpy.arange(indexes.start, indexes.stop)
        self._validate_source(source)
        rval = []
        for so in source:
            batch = self.map_functions[self.data_specs[1].index(so)](indexes)
            batch_buffer = self.batch_buffers[self.data_specs[1].index(so)]
            dim = self.data_specs[0].components[self.data_specs[1].index(so)].dim
            if batch_buffer is None or batch_buffer.shape != (len(batch), dim):
                batch_buffer = numpy.zeros((len(batch), dim),
                                           dtype=batch[0].dtype)
            for i, example in enumerate(batch):
                batch_buffer[i] = example
            rval.append(batch_buffer)
        return tuple(rval)

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 rng=None, data_specs=None, return_tuple=False, convert=None):
        """
        .. todo::

            WRITEME
        """
        if data_specs is None:
            data_specs = self._iter_data_specs

        # If there is a view_converter, we have to use it to convert
        # the stored data for "features" into one that the iterator
        # can return.
        space, source = data_specs
        if isinstance(space, CompositeSpace):
            sub_spaces = space.components
            sub_sources = source
        else:
            sub_spaces = (space,)
            sub_sources = (source,)

        if convert == None:
            if hasattr(self, '_iter_convert'):
                convert = self._iter_convert
            else:
                convert = []
                for sp, src in safe_zip(sub_spaces, sub_sources):
                    convert.append(None)

        # TODO: Refactor
        if mode is None:
            if hasattr(self, '_iter_subset_class'):
                mode = self._iter_subset_class
            else:
                raise ValueError('iteration mode not provided and no default '
                                 'mode set for %s' % str(self))
        else:
            mode = resolve_iterator_class(mode)

        if batch_size is None:
            batch_size = getattr(self, '_iter_batch_size', None)
        if num_batches is None:
            num_batches = getattr(self, '_iter_num_batches', None)
        if rng is None and mode.stochastic:
            rng = self.rng
        return FiniteDatasetIterator(self,
                                     mode(self.num_examples, batch_size,
                                          num_batches, rng),
                                     data_specs=data_specs,
                                     return_tuple=return_tuple,
                                     convert=convert)

if __name__ == "__main__":
    # train_timit = TIMIT("train", frame_length=240, overlap=10,
    #                     frames_per_example=5)
    valid_timit = TIMIT("valid", frame_length=1, overlap=0,
                        frames_per_example=100, audio_only=False)
    # test_timit = TIMIT("test", frame_length=240, overlap=10,
    #                     frames_per_example=5)
    # import pdb; pdb.set_trace()
    it = valid_timit.iterator(mode='random_uniform', num_batches=100, batch_size=256)
    import pdb; pdb.set_trace()
    for (f, t) in it:
        print f.shape
