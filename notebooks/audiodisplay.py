# This file was written by Brian E. Granger
# (https://github.com/ellisonbg/talk-sicm2-2013) and is licensed under
# the MIT License.

import os
import struct
from io import BytesIO
import base64
import sys

from IPython.utils.py3compat import string_types
from IPython.core.display import DisplayObject

class Audio(DisplayObject):
    def __init__(self, data=None, url=None, filename=None, rate=None, autoplay=False):
        """Create an audio player given raw PCM data.

        When this object is returned by an expression or passed to the
        display function, it will result in an audio player widget
        displayed in the frontend.

        Parameters
        ----------
        data : unicode, str or bytes
            The raw audio data.
        url : unicode
            A URL to download the data from.
        filename : unicode
            Path to a local file to load the data from.
        rate : integer
            The sampling rate of the raw data.
            Only required when data parameter is being used
        autoplay : bool
            Set to True if the audio should immediately start playing.

            Default is `False`.

        Examples
        --------
        # embedded raw audio data
        Audio(data=2**13*numpy.sin(2*pi*440/44100*numpy.arange(44100)).astype(numpy.int16),
            rate=44100)

        # Specifying Audio(url=...) or Audio(filename=...) will load the data
        # from an existing WAV file and embed it into the notebook.
        Audio(filename='something.wav', autoplay=True)

        """

        super(Audio, self).__init__(data, url, filename)

        if data is not None and not isinstance(data, string_types):
            buffer = BytesIO()
            buffer.write(b'RIFF')
            buffer.write(b'\x00\x00\x00\x00')
            buffer.write(b'WAVE')

            buffer.write(b'fmt ')
            if data.ndim == 1:
                noc = 1
            else:
                noc = data.shape[1]
            bits = data.dtype.itemsize * 8
            sbytes = rate*(bits // 8)*noc
            ba = noc * (bits // 8)
            buffer.write(struct.pack('<ihHIIHH', 16, 1, noc, rate, sbytes, ba, bits))

            # data chunk
            buffer.write(b'data')
            buffer.write(struct.pack('<i', data.nbytes))

            if data.dtype.byteorder == '>' or (data.dtype.byteorder == '=' and sys.byteorder == 'big'):
                data = data.byteswap()

            buffer.write(data.tostring())
            size = buffer.tell()
            buffer.seek(4)
            buffer.write(struct.pack('<i', size-8))

            self.data = buffer.getvalue()


        self.autoplay = autoplay

    def _repr_html_(self):
        src = """
        <audio controls="controls" {autoplay}>
          <source controls src="{src}" type="audio/wav" />
          Your browser does not support the audio element.
        </audio>
        """.format(src=self.src_attr(),autoplay=self.autoplay_attr())
        return src

    def src_attr(self):
        if(self.data is not None):
            return """data:audio/wav;base64,{base64}""".format(base64=base64.encodestring(self.data))
        else:
            return ""

    def autoplay_attr(self):
        if(self.autoplay):
            return 'autoplay="autoplay"'
        else:
            return ''
