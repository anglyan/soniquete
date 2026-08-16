"""Plot the different envelope shapes available for a Waveform.

A Waveform is a sum of pure tones shaped by one or more Envelopes. This
example builds the same A4 tone with each of the built-in envelope shapes
and plots the resulting waveforms side by side.
"""

import numpy as np
import matplotlib.pyplot as pt

from soniquete.shapes import (
    ExponentialEnvelope,
    FadeEnvelope,
    GaussianEnvelope,
    HammingEnvelope,
    SineModulatedEnvelope,
)

DURATION = 1.0

envelopes = [GaussianEnvelope(DURATION), 
    ExponentialEnvelope(DURATION),
    SineModulatedEnvelope(DURATION),
    HammingEnvelope(DURATION),
    FadeEnvelope(DURATION, rise_time=0.1)]

t = np.arange(0, DURATION, 0.001)
wave = np.ones_like(t)

for i, en in enumerate(envelopes):
    s = en(wave, sample_rate=1000)
    pt.plot(t, 1.1+s, label=en.name)

pt.legend()
pt.tight_layout()
pt.show()


