from paraview.simple import *

class ParaViewPlotter:
    def __init__(self, filename, background_color='white', show_edges=True, save_image_path=None):
        self.filename = filename
        self.background_color = background_color
        self.show_edges = show_edges
        self.save_image_path = save_image_path

    def plot(self):
        
        # 加载数据
        data = XMLUnstructuredGridReader(FileName=self.filename)
        
        # 显示数据
        display = Show(data)
        display.Representation = 'Surface With Edges' if self.show_edges else 'Surface'
        
        # 设置背景色
        view = GetActiveViewOrCreate('RenderView')
        view.Background = self.background_color
        
        # 渲染视图
        Render()
        
        # 如果指定了保存图片的路径，则保存图片
        if self.save_image_path:
            SaveScreenshot(self.save_image_path)
        
        # 开启交互模式
        Interact()

