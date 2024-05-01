from fealpy.plotter.gl import OCAMModel


# 读取输入图像
imgname = sys.argv[1]
outname = imgname.split('/')[-1][:-4]
img = cv2.imread(imgname)

cam = OCAMModel(1, 1, 1, 1, 1, 1, 1, 1, 1, 1920, 1080, 1500, 1500)

# 进行透视矫正
result = cam.unwarp(img)

# 显示结果
cv2.imwrite(outname+'ccc.jpg', result)



