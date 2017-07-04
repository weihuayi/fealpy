import os
import numpy as np
import natsort
import vtk

import sys

dataPath = sys.argv[1]
outputfile = sys.argv[2]
for dirName, subDir, fileList in os.walk(dataPath):
    pass    
index = natsort.index_natsorted(fileList)
fileList = natsort.order_by_index(fileList, index)

stringArray = vtk.vtkStringArray()
for i, fileName in enumerate(fileList):
    stringArray.InsertNextValue(fileName)

reader = vtk.vtkDICOMImageReader()
reader.SetDirectoryName(dataPath)
reader.SetFileNames(stringArray)
writer = vtk.vtkMetaImageWriter()
writer.SetInputConnection(reader.GetOutputPort())
writer.SetFileName(outputfile + '.mhd')
writer.Write()
