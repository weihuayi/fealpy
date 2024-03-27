import glfw
from OpenGL.GL import *
import numpy as np

class OpenGLPlotter:
    def __init__(self, width=800, height=600, title="OpenGL Application"):
        if not glfw.init():
            raise Exception("GLFW cannot be initialized!")
        
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

        void main()
        {
            gl_Position = vec4(aPos, 1.0);
        }
        """
        self.fragment_shader_source = """
        #version 330 core
        out vec4 FragColor;

        void main()
        {
            FragColor = vec4(1.0, 1.0, 1.0, 1.0);
        }
        """

        # 编译着色器
        self.shader_program = self.create_shader_program()

        self.VAO = None
        self.VBO = None
        self.EBO = None

        glfw.set_key_callback(self.window, self.key_callback)
        glfw.set_cursor_pos_callback(self.window, self.mouse_callback)

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
            glClearColor(0.2, 0.3, 0.3, 1.0)

            # 使用着色器程序
            glUseProgram(self.shader_program)
            glBindVertexArray(self.VAO)
            glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)  # 绘制三角形

            glfw.swap_buffers(self.window)

        glfw.terminate()

    def key_callback(self, window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)

    def mouse_callback(self, window, xpos, ypos):
        print(f"Mouse position: {xpos}, {ypos}")

def main():
    # 假设nodes和cells是你的网格数据
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    cells = np.array([[0, 1, 2]])

    plotter = OpenGLPlotter()
    plotter.load_mesh(nodes, cells)
    plotter.run()

if __name__ == "__main__":
    main()

