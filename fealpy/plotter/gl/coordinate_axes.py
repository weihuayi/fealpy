import numpy as np
from OpenGL.GL import *

class CoordinateAxes:
   def __init__(self):
       self.vertex_shader_source = """
       #version 460 core
       layout (location = 0) in vec3 position;
       layout (location = 1) in vec3 color;
       out vec3 ourColor;
       uniform mat4 projection;
       uniform mat4 view;
       uniform mat4 model;
       void main() {
           gl_Position = projection * view * model * vec4(position, 1.0f);
           ourColor = color;
       }
       """
       
       self.fragment_shader_source = """
       #version 460 core
       in vec3 ourColor;
       out vec4 color;
       void main() {
           color = vec4(ourColor, 1.0f);
       }
       """
       self.shader_program = self.create_shader_program()
       self.initialize()

   def initialize(self):
       # 坐标轴顶点和颜色数据
       self.vertices = np.array([
           0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  # X轴（红色）
           1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 1.0, 0.0,  # Y轴（绿色）
           0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0, 1.0,  # Z轴（蓝色）
           0.0, 0.0, 1.0, 0.0, 0.0, 1.0
       ], dtype=np.float32)
       self.vao = glGenVertexArrays(1)
       self.vbo = glGenBuffers(1)
       glBindVertexArray(self.vao)
       glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
       glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
       # 位置属性
       glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * self.vertices.itemsize, ctypes.c_void_p(0))
       glEnableVertexAttribArray(0)
       # 颜色属性
       glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * self.vertices.itemsize, ctypes.c_void_p(3 * self.vertices.itemsize))
       glEnableVertexAttribArray(1)
       glBindBuffer(GL_ARRAY_BUFFER, 0)
       glBindVertexArray(0)

   def create_shader_program(self):
       # 编译顶点和片段着色器
       vertex_shader = self.compile_shader(self.vertex_shader_source, GL_VERTEX_SHADER)
       fragment_shader = self.compile_shader(self.fragment_shader_source, GL_FRAGMENT_SHADER)
       # 链接着色器程序
       program = glCreateProgram()
       glAttachShader(program, vertex_shader)
       glAttachShader(program, fragment_shader)
       glLinkProgram(program)
       # 检查链接错误
       if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
           raise RuntimeError(glGetProgramInfoLog(program))
       glDeleteShader(vertex_shader)
       glDeleteShader(fragment_shader)
       return program

   def compile_shader(self, source, shader_type):
       shader = glCreateShader(shader_type)
       glShaderSource(shader, source)
       glCompileShader(shader)
       if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
           error = glGetShaderInfoLog(shader).decode('utf-8')
           raise RuntimeError(f"Shader compile error: {error}")
       return shader

   def render(self, projection, view, model):
       glUseProgram(self.shader_program)

       # 这里是直接传递相同的投影矩阵和视图矩阵，它们用于场景中的其他对象
       glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "projection"), 1, GL_FALSE, projection)
       glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "view"), 1, GL_FALSE, view)
       glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "model"), 1, GL_FALSE, model)

       glBindVertexArray(self.vao)
       glDrawArrays(GL_LINES, 0, 6)
       glBindVertexArray(0)
       glUseProgram(0)  # 还原使用的着色器程序，以避免影响场景中的其他渲染

