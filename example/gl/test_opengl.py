import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
from PIL import Image
import ctypes
import glm

from fealpy.mesh import TriangleMesh
from fealpy.plotter.gl import Camera

# 顶点着色器
vertex_src = """
# version 330
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_texture;

uniform mat4 view;
uniform mat4 projection;

out vec2 v_texture;
void main()
{
    gl_Position = projection * view * vec4(a_position, 1.0);
    v_texture = a_texture;
}
"""

# 片段着色器
fragment_src = """
# version 330
in vec2 v_texture;
out vec4 out_color;
uniform sampler2D s_texture;
void main()
{
    out_color = texture(s_texture, v_texture);
}
"""

def load_texture(path):
    img = Image.open(path)
    img = img.transpose(Image.FLIP_TOP_BOTTOM) # 将图片上下翻转，因为OpenGL的纹理坐标和图片的默认坐标是反的
    img_data = img.convert("RGBA").tobytes() # 转换图片为RGBA格式，并转换为字节
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
    glGenerateMipmap(GL_TEXTURE_2D)
    return texture

def mouse_callback(window, xpos, ypos):
    global camera
    camera.process_mouse_movement(xpos, ypos)

def main():

    global camera
    camera = Camera()
    # 初始化 GLFW
    if not glfw.init():
        return

    window = glfw.create_window(800, 600, "OpenGL Window", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.set_cursor_pos_callback(window, mouse_callback)
    glClearColor(0.2, 0.3, 0.3, 1.0)

    # 创建着色器程序
    shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                            compileShader(fragment_src, GL_FRAGMENT_SHADER))


    """
    # 定义顶点数据和UV坐标
    vertices = np.array([
        -0.5, -0.5, 0.0,  0.0, 0.0,  # 左下角
         0.5, -0.5, 0.0,  1.0, 0.0,  # 右下角
         0.5,  0.5, 0.0,  1.0, 1.0,  # 右上角
        -0.5,  0.5, 0.0,  0.0, 1.0   # 左上角
    ], dtype=np.float32)

    indices = np.array([
        0, 1, 2,
        2, 3, 0
    ], dtype=np.uint32)

    """
    mesh, U, V = TriangleMesh.from_ellipsoid_surface(10, 80, radius=(4, 2, 1),
            theta=(np.pi/2, np.pi/2+np.pi/3))
    node = mesh.entity('node')
    cell = mesh.entity('cell')
    vertices = np.hstack((node, U.flatten().reshape(-1, 1), V.flatten().reshape(-1, 1)), dtype=np.float32)
    indices = np.array(cell, dtype=np.uint32)

    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    # 位置属性
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    # 纹理坐标属性
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)

    # 加载并绑定纹理
    texture = load_texture("/home/why/we.jpg")

    projection = glm.perspective(glm.radians(45.0), 800 / 600, 0.1, 100.0)
    proj_loc = glGetUniformLocation(shader, "projection")
    glUseProgram(shader)
    glUniform1i(glGetUniformLocation(shader, "s_texture"), 0)  # 将纹理绑定到着色器的纹理单元
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, glm.value_ptr(projection))


    # 渲染循环
    while not glfw.window_should_close(window):
        glfw.poll_events()

        # 清除颜色和深度缓冲区
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(shader)

        # 更新并使用摄像机的视图矩阵
        view = camera.get_view_matrix()
        view_loc = glGetUniformLocation(shader, "view")
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view))

        # 绑定VAO以绘制图形
        glBindVertexArray(VAO)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

        glfw.swap_buffers(window)

    # 清理资源
    glDeleteVertexArrays(1, [VAO])
    glDeleteBuffers(1, [VBO])
    glDeleteBuffers(1, [EBO])
    glDeleteTextures(1, [texture])
    glfw.terminate()

if __name__ == "__main__":
    main()
