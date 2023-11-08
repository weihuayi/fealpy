from paraview.simple import *

data = XMLUnstructuredGridReader(FileName='test.vtu')
Show(data)
Render()
Interact()
