# soniquete

## About

`soniquete` is a simple module to generate, mix, and reproduce waveforms.

## Building blocks

`soniquete` implements the following building blocks:

- `Tone` objects, representing a pure single frequency wave with a certain amplitude and phase.

- `Waveform` objects contain a superposition of tones shaped by one or more envelopes and compute the
corresponding `numpy` arrays.

- `Block` objects implement arbitrary
sounds from `numpy` arrays.

- `Envelope` objects shape the intensity of any array. `soniquete` implements different types of envelopes.


## Examples

```python

import soniquete as sq
import numpy as np

f = sq.Frequency('C2')

t = np.arange(0, 0.5, 1.0/sq.dsr)
sound = np.sin(2 * np.pi * f.hz * t)

block = sq.Block(sound, sample_rate=sq.DSR)

print("Playing block...")
block.play()

```

