# soniquete

## About

Soniquete is a simple module to generate, mix, and reproduce waveforms.

## Examples

```python

import soniquete as sq
import numpy as np

f = sq.Frequency('C2')

t = np.arange(0, 0.5, 1.0/sq.dsr)
sound = np.sin(2 * np.pi * f.hz * t)

block = sq.Block(sound, sample_rate=sq.dsr)

print("Playing block...")
block.play()

```

