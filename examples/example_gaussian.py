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
DURATION = 2.0
NFREQ = 100

f0 = Frequency('C3').hz
sigma = 0.02

flist = []

for i in range(NFREQ):
    fi = f0 + 3*sigma*f0*(np.random.random()-0.5)
    ai = np.exp(-((fi-f0)/(sigma*f0))**2)
    print(fi, ai)
    flist.append(Tone(fi, ai))

w1 = Waveform(flist, duration=DURATION,
    envelopes=[ExponentialEnvelope(DURATION, decay_time=0.2)])

block = sq.Block(w1.array)

#pt.plot(w1.t, w1.array)
#pt.show()

block.play()

