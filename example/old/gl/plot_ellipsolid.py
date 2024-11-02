import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PIL import Image
import ipdb

# 定义椭球体的顶点数目
slices = 50
stacks = 50
window = 0
rotation_angle = 0.0

# 加载纹理
def load_texture(filename):
    img = Image.open(filename)
    #img.show()
    img = img.convert("RGB")
    img_data = np.array(list(img.getdata()), np.uint8)
    texture_id = glGenTextures(1)

    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    return texture_id

# 创建椭球体顶点数据
def draw_ellipsoid(radius_x, radius_y, radius_z):
    for j in range(stacks):
        glBegin(GL_QUAD_STRIP)
        for i in range(slices + 1):
            theta = (j / stacks) * np.pi
            phi = (i / slices) * 2 * np.pi
            x = radius_x * np.sin(theta) * np.cos(phi)
            y = radius_y * np.sin(theta) * np.sin(phi)
            z = radius_z * np.cos(theta)
            u = i / slices
            v = j / stacks
            glTexCoord2f(u, v)
            glVertex3f(x, y, z)

            theta_next = ((j + 1) / stacks) * np.pi
            x_next = radius_x * np.sin(theta_next) * np.cos(phi)
            y_next = radius_y * np.sin(theta_next) * np.sin(phi)
            z_next = radius_z * np.cos(theta_next)
            v_next = (j + 1) / stacks
            glTexCoord2f(u, v_next)
            glVertex3f(x_next, y_next, z_next)
        glEnd()

# 渲染场景
def render_scene():
    global rotation_angle

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(0.0, 0.0, -5)
    glRotatef(rotation_angle, 1, 1, 1)  # 绕着 (1, 1, 1) 方向旋转
    rotation_angle += 0.5  # 每次增加 0.5 度

    glBindTexture(GL_TEXTURE_2D, texture_id)
    draw_ellipsoid(1, 1.5, 2)  # 修改椭球体的半径参数

    glutSwapBuffers()

# 主函数
def main():
    global texture_id
    global window

    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    window = glutCreateWindow(b"OpenGL Ellipsoid with Texture")

    glEnable(GL_TEXTURE_2D)
    texture_id = load_texture('/home/why/we.jpg')

    glutDisplayFunc(render_scene)
    glutIdleFunc(render_scene)

    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (800 / 600), 0.1, 50.0)

    glutMainLoop()

if __name__ == "__main__":
    main()
