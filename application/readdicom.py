import dicom
import os
import numpy as np
import natsort
import  matplotlib.pyplot as plt
from matplotlib import  cm

dataPath = "/home/why/se0"
for dirName, subDir, fileList in os.walk(dataPath):
    pass    
index = natsort.index_natsorted(fileList)
fileList = natsort.order_by_index(fileList, index)

data0 = dicom.read_file(os.path.join(dirName, fileList[0]))

# Load dimensions based on the number of rows, columns, and slices
constPixelDims = (int(data0.Rows), int(data0.Columns), len(fileList))

# Load spacing values
constPixelSpacing = (
        float(data0.PixelSpacing[0]), 
        float(data0.PixelSpacing[1]),
        float(data0.SliceThickness))
x = np.arange(
        0.0, 
        (constPixelDims[0]+1)*constPixelSpacing[0],
        constPixelSpacing[0])
y = np.arange(
        0.0, 
        (constPixelDims[1]+1)*constPixelSpacing[1],
        constPixelSpacing[1])
z = np.arange(
        0.0,
        (constPixelDims[2]+1)*constPixelSpacing[2],
        constPixelSpacing[2])

# The array is sized based on 'constPixelDims'
data = np.zeros(constPixelDims, dtype=data0.pixel_array.dtype)

# Loop through all the DICOM files
for i, fileName in enumerate(fileList):
    ds = dicom.read_file(os.path.join(dirName, fileName))
    data[:, :, i] = ds.pixel_array 

plt.figure(dpi=300)
plt.axes().set_aspect('equal', 'datalim')
plt.set_cmap(plt.gray())
plt.pcolormesh(x, y, np.flipud(data[:, :, 200]))
plt.show()

