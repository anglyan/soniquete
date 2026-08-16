import soniquete as sq

import numpy as np
import matplotlib.pyplot as pt

f = sq.Frequency('C2')

t = np.arange(0, 0.5, sq.DDT)
sound = np.sin(2 * np.pi * f.hz * t)

block = sq.Block(sound, sample_rate=sq.DSR)

print("Playing block...")
block.play()
print("Comparing block to array...")

pt.plot(t, block.array, label="Block")
pt.plot(t, 1.5+sound, label="Array + 1.5")
pt.legend()
pt.show()
