import cv2
import sys

from fealpy.plotter.gl import OCAMModel


# 读取输入图像
imgname = sys.argv[1]
outname = imgname.split('/')[-1][:-4]

cam = OCAMModel(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, sys.argv[2])

# 进行透视矫正
result = cam.undistort_chess(imgname)
result = cam.perspective(result)

# 显示结果
cv2.imwrite(outname+'ccc.jpg', result)



