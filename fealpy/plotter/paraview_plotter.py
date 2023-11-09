from paraview.simple import *

class ParaViewPlotter:
    def __init__(self, args):
        self.args = args

    def plot(self):
        """
        """

        args = self.args
        # 加载数据
        data = XMLUnstructuredGridReader(FileName=args.filename)
        
        # 显示数据
        display = Show(data)
        display.Representation = args.show_type 
        
        # 设置背景色
        view = GetActiveViewOrCreate('RenderView')
        view.Background = args.background_color
        
        # 渲染视图
        Render()
        
        # 如果指定了保存图片的路径，则保存图片
        #if self.save_image_path:
        #    SaveScreenshot(self.save_image_path)
        
        # 开启交互模式
        Interact()

