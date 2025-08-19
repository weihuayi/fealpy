
import argparse
import matplotlib.pyplot as plt

from fealpy.backend import bm 

from fealpy.geometry import DLDMicrofluidicChipModeler


options = {}

modeler = DLDMicrofluidicChipModeler(options)
modeler.build()

fig = plt.figure()
ax = fig.add_subplot(111)
modeler.add_plot(ax)
plt.show()



