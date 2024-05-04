import glfw
from OpenGL.GL import *

# 初始化GLFW
if not glfw.init():
    raise Exception("GLFW cannot be initialized!")

# 创建一个窗口，但在这里不显示它
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
window = glfw.create_window(640, 480, "Hidden Window", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window cannot be created!")

glfw.make_context_current(window)

# 获取OpenGL的版本
version = glGetString(GL_VERSION)
renderer = glGetString(GL_RENDERER)
vendor = glGetString(GL_VENDOR)

print(f"OpenGL version: {version}")
print(f"Renderer: {renderer}")
print(f"Vendor: {vendor}")

glfw.terminate()
