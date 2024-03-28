import glfw
from OpenGL.GL import *
import numpy as np

"""
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(name)s: %(message)s', datefmt='%m-%d %H:%M:%S')
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)
    logger.propagate = False
"""

from fealpy import logger

class OpenGLPlotter:
    def __init__(self, width=800, height=600, title="OpenGL Application"):
        if not glfw.init():
            raise Exception("GLFW cannot be initialized!")

        self.show_edges = True  # 默认显示网格线
        self.last_mouse_pos = (width / 2, height / 2)
        self.first_mouse_use = True

        self.mode = 2  # 默认同时显示边和面
        self.faceColor = np.array([0.5, 0.7, 0.9, 1.0], dtype=np.float32)  # 浅蓝色
        self.edgeColor = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)  # 白色
        self.bgColor = np.array([0.1, 0.2, 0.3, 1.0], dtype=np.float32)   # 深海军蓝色背景
        
        # 设置使用OpenGL核心管线
        #glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        #glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        #glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        #glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        
        # 确保title是字符串类型，然后在这里对其进行编码
        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window cannot be created!")
        
        glfw.make_context_current(self.window)

        # 设置视口大小
        glViewport(0, 0, width, height)

        # 着色器源码
        self.vertex_shader_source = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        uniform mat4 transform; //变换矩阵

        void main()
        {
            gl_Position = transform * vec4(aPos, 1.0);
        }
        """
        self.fragment_shader_source = """
        #version 330 core
        uniform int mode;  // 0: 显示面，1: 显示边，2: 显示面和边
        uniform vec4 faceColor;
        uniform vec4 edgeColor;
        out vec4 FragColor;

        void main()
        {
            if (mode == 0) {
                FragColor = faceColor;  // 只显示面
            } else if (mode == 1) {
                FragColor = edgeColor;  // 只显示边
            } else if (mode == 2) {
                // 这个逻辑取决于您是如何组织网格数据的
                // 例如，可以根据gl_FragCoord是否在边上来决定颜色
                // 这里只是一个概念示例
                FragColor = faceColor;  // 同时显示面和边
            }
        }
        """

        # 编译着色器
        self.shader_program = self.create_shader_program()

        self.VAO = None
        self.VBO = None
        self.EBO = None

        glfw.set_key_callback(self.window, self.key_callback)
        glfw.set_cursor_pos_callback(self.window, self.mouse_callback)

        self.transform = np.identity(4, dtype=np.float32)
        glfw.set_scroll_callback(self.window, self.scroll_callback)

    def load_mesh(self, nodes, cells):
        vertices = nodes[cells].reshape(-1, nodes.shape[1])
        self.vertex_count = len(vertices)

        # 创建并绑定VAO
        if self.VAO is None:
            self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        # 创建并绑定VBO
        if self.VBO is None:
            self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        # 设置顶点属性指针
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertices.itemsize * 3, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # 解绑VBO和VAO
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def compile_shader(self, source, shader_type):
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
            error = glGetShaderInfoLog(shader).decode('utf-8')
            raise Exception(f"Shader compile failure: {error}")
        return shader

    def create_shader_program(self):
        vertex_shader = self.compile_shader(self.vertex_shader_source, GL_VERTEX_SHADER)
        fragment_shader = self.compile_shader(self.fragment_shader_source, GL_FRAGMENT_SHADER)
        shader_program = glCreateProgram()
        glAttachShader(shader_program, vertex_shader)
        glAttachShader(shader_program, fragment_shader)

        # 在链接之前绑定FragColor的位置
        # glBindFragDataLocation(shader_program, 0, "FragColor")

        glLinkProgram(shader_program)
        if glGetProgramiv(shader_program, GL_LINK_STATUS) != GL_TRUE:
            error = glGetProgramInfoLog(shader_program).decode('utf-8')
            raise Exception(f"Program link failure: {error}")

        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)

        return shader_program

    def run(self):
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            
            # 清除颜色缓冲区
            glClear(GL_COLOR_BUFFER_BIT)
            glClearColor(
                    self.bgColor[0], 
                    self.bgColor[1],
                    self.bgColor[2],
                    self.bgColor[3])

            # 更新颜色和模式
            glUniform4fv(glGetUniformLocation(self.shader_program, "faceColor"), 1, self.faceColor)
            glUniform4fv(glGetUniformLocation(self.shader_program, "edgeColor"), 1, self.edgeColor)
            glUniform1i(glGetUniformLocation(self.shader_program, "mode"), self.mode)

            # 使用着色器程序
            glUseProgram(self.shader_program)

            # 更新着色器的uniform变量以应用变换
            # 获取uniform变量的位置，并检查它们是否有效
            face_color_location = glGetUniformLocation(self.shader_program, "faceColor")
            edge_color_location = glGetUniformLocation(self.shader_program, "edgeColor")
            mode_location = glGetUniformLocation(self.shader_program, "mode")
            transform_location = glGetUniformLocation(self.shader_program, "transform")

            if face_color_location == -1 or edge_color_location == -1 or mode_location == -1 or transform_location == -1:
                logger.error("One or more uniform locations are invalid.")
                continue

            # 更新着色器的uniform变量
            glUniform4fv(face_color_location, 1, self.faceColor)
            glUniform4fv(edge_color_location, 1, self.edgeColor)
            glUniform1i(mode_location, self.mode)
            glUniformMatrix4fv(transform_location, 1, GL_FALSE, self.transform)

            glBindVertexArray(self.VAO)

            # 根据show_edges变量决定是否绘制网格线
            if self.show_edges:
                # 绘制网格线的代码
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)  # 绘制线框
                glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)  # 恢复默认模式
            else:
                # 只绘制面的代码
                glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)

            glfw.swap_buffers(self.window)

        glfw.terminate()

    def key_callback(self, window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)

        translate_speed = 0.1
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_UP:  # 向上平移
                self.transform[3, 1] += translate_speed
            elif key == glfw.KEY_DOWN:  # 向下平移
                self.transform[3, 1] -= translate_speed
            elif key == glfw.KEY_RIGHT:  # 向右平移
                self.transform[3, 0] += translate_speed
            elif key == glfw.KEY_LEFT:  # 向左平移
                self.transform[3, 0] -= translate_speed

        if key == glfw.KEY_E and action == glfw.PRESS:
            self.show_edges = not self.show_edges  # 切换网格线的显示状态

        logger.debug("Translating: {}".format(key))

    def mouse_callback(self, window, xpos, ypos):
        print(f"Mouse position: {xpos}, {ypos}")
        if self.first_mouse_use:
            self.last_mouse_pos = (xpos, ypos)
            self.first_mouse_use = False

        xoffset = xpos - self.last_mouse_pos[0]
        yoffset = self.last_mouse_pos[1] - ypos  # 注意这里的y方向与屏幕坐标系相反
        self.last_mouse_pos = (xpos, ypos)

        sensitivity = 0.1
        xoffset *= sensitivity
        yoffset *= sensitivity

        # 生成旋转矩阵
        # 这里简化处理，只根据xoffset和yoffset来做基本的旋转，实际应用中可能需要更复杂的旋转逻辑
        rotation_x = np.array([[1, 0, 0, 0],
                               [0, np.cos(yoffset), -np.sin(yoffset), 0],
                               [0, np.sin(yoffset), np.cos(yoffset), 0],
                               [0, 0, 0, 1]], dtype=np.float32)

        rotation_y = np.array([[np.cos(xoffset), 0, np.sin(xoffset), 0],
                               [0, 1, 0, 0],
                               [-np.sin(xoffset), 0, np.cos(xoffset), 0],
                               [0, 0, 0, 1]], dtype=np.float32)

        self.transform = np.dot(self.transform, rotation_x)
        self.transform = np.dot(self.transform, rotation_y)

        logger.debug("Rotating: X offset {}, Y offset {}".format(xoffset, yoffset))

    def scroll_callback(self, window, xoffset, yoffset):
        """鼠标滚轮回调函数，用于缩放视图。"""
        scale_factor = 1.1  # 缩放系数
        if yoffset < 0:  # 向下滚动，缩小
            scale_factor = 1.0 / scale_factor
        # 更新变换矩阵
        self.transform[:3, :3] *= scale_factor
        logger.debug("Zooming: {}".format("In" if scale_factor > 1 else "Out"))

def main():
    # 假设nodes和cells是你的网格数据
    from fealpy.mesh import TriangleMesh

    mesh = TriangleMesh.from_ellipsoid_surface(10, 100, 
            radius=(4, 2, 1), theta=(np.pi/2, np.pi/2+np.pi/3))
    node = mesh.entity('node')
    cell = mesh.entity('cell')
    nodes = np.array(node, dtype=np.float32)
    cells = np.array(cell, dtype=np.uint32)

    plotter = OpenGLPlotter()
    plotter.load_mesh(nodes, cells)
    plotter.run()

if __name__ == "__main__":
    main()

