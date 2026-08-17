"""Plot the different envelope shapes available for a Waveform.

A Waveform is a sum of pure tones shaped by one or more Envelopes. This
example builds the same A4 tone with each of the built-in envelope shapes
and plots the resulting waveforms side by side.
"""

import numpy as np
import matplotlib.pyplot as pt

from soniquete import Waveform, Frequency, Tone
from soniquete.shapes import ExponentialEnvelope

import soniquete as sq
import numpy as np
DURATION = 4.0
NFREQ = 100

f0 = Frequency('C4').hz
sigma = 0.3

flist = []

for i in range(NFREQ):

    offset = 6*sigma*(np.random.random()-0.5)
    fi = Frequency('C2', offset=offset)
    ai = np.exp(-(offset/sigma)**2)
    phasei = 2*np.pi*np.random.random()
    print(fi, ai)
    flist.append(Tone(fi, ai, phasei))

for i in range(NFREQ):

    offset = 6*sigma*(np.random.random()-0.5)
    fi = Frequency('F1', offset=offset)
    ai = np.exp(-(offset/sigma)**2)
    phasei = 2*np.pi*np.random.random()
    print(fi, ai)
    flist.append(Tone(fi, ai, phasei))


w1 = Waveform(flist, duration=DURATION,
    envelopes=[ExponentialEnvelope(DURATION, 0.2*DURATION)])

block = sq.Block(w1.array)

block.play()
pt.plot(w1.t, block.array)
pt.show()
