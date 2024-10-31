import glfw
from OpenGL.GL import *
import numpy as np
from quaternion import Quaternion
from arc_ball import ArcBall

class OpenGLApplication:
    def __init__(self, width, height, title):
        self.width = width
        self.height = height
        self.title = title
        self.init_glfw()
        self.shader_program = self.create_shader_program()
        self.arcball = ArcBall(0, 0, width, height)
        self.rotation = Quaternion(np.array([0, 0, 1, 0]))
        self.rotation_loc = None
        self.setup_callbacks()
        self.setup_triangle()  # 设置三角形

    def init_glfw(self):
        if not glfw.init():
            raise Exception("Failed to initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

        self.window = glfw.create_window(self.width, self.height, self.title, None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Failed to create GLFW window")

        glfw.make_context_current(self.window)

    def create_shader_program(self):
        vertex_src = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        uniform mat4 rotation;
        void main()
        {
            gl_Position = rotation * vec4(aPos, 1.0);
        }
        """
        fragment_src = """
        #version 330 core
        out vec4 FragColor;
        void main()
        {
            FragColor = vec4(1.0, 0.5, 0.2, 1.0);
        }
        """
        vertex_shader = self.compile_shader(vertex_src, GL_VERTEX_SHADER)
        fragment_shader = self.compile_shader(fragment_src, GL_FRAGMENT_SHADER)

        program = glCreateProgram()
        glAttachShader(program, vertex_shader)
        glAttachShader(program, fragment_shader)
        glLinkProgram(program)
        if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
            error = glGetProgramInfoLog(program).decode()
            print(f"Shader link failed:\n{error}")

        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
        return program

    def compile_shader(self, source, shader_type):
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
            error = glGetShaderInfoLog(shader).decode()
            print(f"Shader compilation failed:\n{error}")
        return shader

    def setup_callbacks(self):
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.cursor_position_callback)

    def mouse_button_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            x, y = glfw.get_cursor_pos(window)
            self.arcball.position = self.arcball.project_to_ball(x - self.width/2.0, self.height/2.0 - y)

    def cursor_position_callback(self, window, x, y):
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS and self.arcball.position is not None:
            q = self.arcball.update(x-self.width/2.0, self.height/2.0 - y)
            self.rotation = q * self.rotation
            rotation_matrix = self.rotation.convert_to_opengl_matrix()
            glUniformMatrix4fv(self.rotation_loc, 1, GL_FALSE, rotation_matrix)

    def setup_triangle(self):
        vertices = np.array([
            -0.5, -0.5, 0.0,
             0.5, -0.5, 0.0,
             0.0,  0.5, 0.0
        ], dtype=np.float32)

        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * vertices.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)  # Unbind VAO
        
    def run(self):
        glUseProgram(self.shader_program)
        self.rotation_loc = glGetUniformLocation(self.shader_program, "rotation")

        initial_rotation = np.identity(4, dtype=np.float32)
        glUniformMatrix4fv(self.rotation_loc, 1, GL_FALSE, initial_rotation)

        while not glfw.window_should_close(self.window):
            glClearColor(0.2, 0.3, 0.3, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)

            glUseProgram(self.shader_program)

            glBindVertexArray(self.VAO)  # Bind VAO of the triangle
            glDrawArrays(GL_TRIANGLES, 0, 3)
            glBindVertexArray(0)  # Unbind VAO

            glfw.swap_buffers(self.window)
            glfw.poll_events()

        glfw.terminate()

if __name__ == "__main__":
    app = OpenGLApplication(800, 600, "OpenGL Window")
    app.run()
