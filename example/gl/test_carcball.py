import glfw
from OpenGL.GL import *
import numpy as np
from quaternion import Quaternion
from carc_ball import CArcBall

arcball = CArcBall(800, 600, 0, 0)

# 更新着色器源代码
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

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        error = glGetShaderInfoLog(shader).decode()
        print(f"Shader compilation failed:\n{error}")
    return shader

def create_shader_program(vertex_src, fragment_src):
    vertex_shader = compile_shader(vertex_src, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_src, GL_FRAGMENT_SHADER)

    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)
    if glGetProgramiv(shader_program, GL_LINK_STATUS) != GL_TRUE:
        error = glGetProgramInfoLog(shader_program).decode()
        print(f"Shader link failed:\n{error}")
    
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    return shader_program

def initialize_window():
    if not glfw.init():
        return None

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    window = glfw.create_window(800, 600, "OpenGL Window", None, None)
    if not window:
        glfw.terminate()
        return None

    glfw.make_context_current(window)
    return window


def mouse_button_callback(window, button, action, mods):
    global arcball
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
        x, y = glfw.get_cursor_pos(window)
        arcball.position = arcball.project_to_ball(x, y)  # 设置arcball的初始位置
    elif button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE:
        arcball.position = None  # 重置位置

def cursor_position_callback(window, x, y):
    global arcball
    if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS and arcball.position is not None:
        p1 = arcball.project_to_ball(x, y)  # 计算新位置
        q = Quaternion.rotation_from_vectors(arcball.position, p1)  # 计算旋转四元数
        rotation_matrix = q.convert_to_opengl_matrix()
        glUniformMatrix4fv(rotation_loc, 1, GL_FALSE, rotation_matrix)
        arcball.position = p1  # 更新当前位置为最新位置

def main():
    window = initialize_window()
    if not window:
        print("Failed to initialize GLFW window.")
        return

    shader_program = create_shader_program(vertex_src, fragment_src)
    rotation_loc = glGetUniformLocation(shader_program, "rotation")

    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)

    vertices = np.array([
        -0.5, -0.5, 0.0,
         0.5, -0.5, 0.0,
         0.0,  0.5, 0.0
    ], dtype=np.float32)

    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * vertices.itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    # 激活着色器程序
    glUseProgram(shader_program)

    # 初始化旋转矩阵为单位矩阵并更新
    initial_rotation = np.identity(4, dtype=np.float32)
    glUniformMatrix4fv(rotation_loc, 1, GL_FALSE, initial_rotation)

    while not glfw.window_should_close(window):
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(shader_program)
        glBindVertexArray(VAO)
        glDrawArrays(GL_TRIANGLES, 0, 3)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glBindVertexArray(0)
    glDeleteVertexArrays(1, [VAO])
    glDeleteBuffers(1, [VBO])
    glfw.terminate()

if __name__ == "__main__":
    main()