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

DURATION = 1.0

f1 = Frequency('C2')
f2 = Frequency('C3')
f3 = Frequency('C4', offset=0.05)


w1 = Waveform([Tone(f1, 0.8), Tone(f2, 0.2), Tone(f3, 0.1)], duration=DURATION,
    envelopes=[ExponentialEnvelope(DURATION, decay_time=0.1)])

block = sq.Block(w1.array)

pt.plot(w1.t, w1.array)
pt.plot(w1.t, block.array)
pt.show()

block.play()

