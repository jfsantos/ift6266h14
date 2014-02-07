from pylearn2.datasets import DenseDesignMatrix
from timit_full import TimitFullCorpusReader
import numpy as np
import itertools

class TimitFrameData(DenseDesignMatrix):
    """
    Dataset for predicting the next acoustic sample.
    """
    def __init__(self, datapath, framelen, overlap, start=0, stop=None, spkrid=None):
        """
        datapath: path to TIMIT raw data (using WAV format)
        framelen: length of the acoustic frames
        overlap: amount of acoustic samples to overlap
        start: index of first frame to use 
        end: index of last frame to use

        FIXME: start and end here are kind of hackish if used with
        spkrid...
        """
        data = TimitFullCorpusReader(datapath)
        # Some list comprehension/zip magic here (but it works!)
        if spkrid:
            utterances = data.utteranceids(spkrid=spkrid)[start:]
        else:
            utterances = data.utteranceids()[start:]
        if stop is not None:
            utterances = utterances[0:stop]
        uttfr = [data.frames(z, framelen, overlap) for z in
                  utterances]
        fr, ph = zip(*[(x[0], x[1]) for x in uttfr])
        fr = np.vstack(fr)*2**-15
        ph = list(itertools.chain(*ph))

        X = fr[:,0:framelen-1]
        y = np.array([fr[:,framelen]]).T # y.ndim has to be 2
        if stop is None:
            stop = len(y)

        super(TimitFrameData,self).__init__(X=X, y=y)

class TimitPhoneData(DenseDesignMatrix):
    """
    Dataset with frames and corresponding one-hot encoded
    phones.
    """
    def __init__(self, datapath, framelen, overlap, start=0, stop=None):
        """
        datapath: path to TIMIT raw data (using WAV format)
        framelen: length of the acoustic frames
        overlap: amount of acoustic samples to overlap
        start: index of first TIMIT file to be used
        end: index of last TIMIT file to be used
        """
        data = TimitFullCorpusReader(datapath)
        # Some list comprehension/zip magic here (but it works!)
        if stop is None:
            utterances = data.utteranceids()[start:]
        else:
            utterances = data.utteranceids()[start:stop]
        spkrfr = [data.frames(z, framelen, overlap) for z in
                  utterances]
        fr, ph = zip(*[(x[0], x[1]) for x in spkrfr])
        X = np.vstack(fr)*2**-15
        ph = list(itertools.chain(*ph))

        # making y a one-hot output
        one_hot = np.zeros((len(ph),len(data.phonelist)),dtype='float32')
        idx = [data.phonelist.index(p) for p in ph]
        for i in xrange(len(ph)):
            one_hot[i,idx] = 1.
        y = one_hot

        super(TimitPhoneData,self).__init__(X=X, y=y)

